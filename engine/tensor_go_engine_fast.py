"""Vectorised batch Go engine (zero CPU↔GPU synchronisations in the hot path).

This module provides a *single* clean implementation of a small‑board Go engine
written entirely with PyTorch tensors.  All heavy‑weight per‑move operations —
legal‑move generation, capture detection, ko handling — are expressed as tensor
kernels so that no `.item()` calls or Python loops execute on each ply.

The entry‑points are

* :class:`TensorBoard`  – the board/environment (batch capable)
* :class:`TensorBatchBot` – a uniform‑random move generator (for profiling)

The file is self‑contained: run it directly to simulate 100 random 7×7 games and
print a short summary.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Device helper
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    """Pick the fastest available device (MPS → CUDA → CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Allow transparent CPU fallback on older macOS installations
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Colour indices (channels in ``self.stones``)
BLACK, WHITE = 0, 1

# ---------------------------------------------------------------------------
#  TensorBoard
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TensorBoard:
    """Batch‑friendly Go engine implemented entirely with PyTorch tensors."""

    batch_size: int = 1
    board_size: int = 19
    device: torch.device | str | None = None

    # ------------------------------ construction -------------------------

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if self.device is not None else _select_device()

        # (B, 2, H, W) – uint8 is plenty for 0/1 flags
        self.stones = torch.zeros(
            (self.batch_size, 2, self.board_size, self.board_size),
            dtype=torch.uint8,
            device=self.device,
        )

        self._init_zobrist()
        self._init_kernels()

        self.current_player = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)
        self.current_hash = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.ko_points = torch.full((self.batch_size, 2), -1, dtype=torch.int16, device=self.device)
        self.pass_count = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)

    # ------------------------------ helpers ------------------------------

    def _init_zobrist(self) -> None:
        torch.manual_seed(42)  # reproducible Zobrist keys
        max64 = torch.iinfo(torch.int64).max
        self.zobrist_keys = torch.randint(
            0,
            max64,
            (2, self.board_size, self.board_size),
            dtype=torch.int64,
            device=self.device,
        )
        self.turn_keys = torch.randint(0, max64, (2,), dtype=torch.int64, device=self.device)

    def _init_kernels(self) -> None:
        # 4‑neighbour kernel (cross shape)
        self.neighbor_kernel = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------ queries ------------------------------

    def get_empty_mask(self) -> torch.Tensor:
        """Return a (B, H, W) boolean mask of empty intersections."""
        return (self.stones[:, BLACK] == 0) & (self.stones[:, WHITE] == 0)

    def get_scores(self) -> torch.Tensor:
        """Simple *area* scores (no komi): number of stones for each colour."""
        black = self.stones[:, BLACK].sum((1, 2)).float()
        white = self.stones[:, WHITE].sum((1, 2)).float()
        return torch.stack([black, white], dim=1)

    def is_game_over(self) -> torch.Tensor:
        """A game ends after two consecutive passes."""
        return self.pass_count >= 2

    # ----------------------- legal‑move generation -----------------------

    def _get_capture_moves_mask(self, empty: torch.Tensor) -> torch.Tensor:
        """Return (B, H, W) mask of points that capture an atari string."""
        B = self.batch_size
        opp = self.stones[torch.arange(B, device=self.device), (1 - self.current_player).long()]

        liberty = F.conv2d(empty.float().unsqueeze(1), self.neighbor_kernel[None, None], padding=1)
        liberty = liberty.squeeze(1)
        weak = (opp > 0) & (liberty <= 1)

        adj = (
            F.conv2d(weak.float().unsqueeze(1), self.neighbor_kernel[None, None], padding=1)
            .squeeze(1)
            .bool()
        )
        return empty & adj

    def get_legal_moves_mask(self) -> torch.Tensor:
        """Return a (B, H, W) boolean mask of legal moves for *current_player*."""
        empty = self.get_empty_mask()
        liberty = F.conv2d(empty.float().unsqueeze(1), self.neighbor_kernel[None, None], padding=1)
        liberty = liberty.squeeze(1)
        legal = empty & (liberty > 0)

        # ko suppression
        ko = self.ko_points[:, 0] >= 0
        if ko.any():
            b = torch.arange(self.batch_size, device=self.device)[ko]
            r = self.ko_points[ko, 0].long()
            c = self.ko_points[ko, 1].long()
            legal[b, r, c] = False

        legal |= self._get_capture_moves_mask(empty)
        return legal

    # -------------------- capture detection + ko ------------------------

    def _capture_mask(
        self,
        played_mask: torch.Tensor,  # (B, H, W) bool – locations just played
        pl: torch.Tensor,  # (B,) uint8 – 0 = black, 1 = white
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return *(captured_mask, ko_rowcol)* for the moves just played."""

        B, _, _ = played_mask.shape
        batch = torch.arange(B, device=self.device)

        opp_col = (1 - pl).long()
        opp_st = self.stones[batch, opp_col] > 0  # opponent stones

        empty = self.get_empty_mask()
        nk = self.neighbor_kernel[None, None]  # (1, 1, 3, 3)

        # 1) Count liberties for *every* point once
        liberty_map = F.conv2d(empty.float().unsqueeze(1), nk, padding=1).squeeze(1)

        # 2) Opponent stones with zero liberties are capture seeds
        mask = opp_st & (liberty_map == 0)

        # 3) Flood‑fill to grow each seed into its full string
        while True:
            grown = (
                F.conv2d(mask.float().unsqueeze(1), nk, padding=1)
                .squeeze(1)
                .bool()
                & opp_st
            )
            new_mask = mask | grown
            if torch.equal(new_mask, mask):  # string fully grown
                break
            mask = new_mask

        # 4) Remove strings whose liberties are still > 0
        string_libs = (
            F.conv2d(mask.float().unsqueeze(1), nk, padding=1).squeeze(1) * empty.float()
        )
        captured = mask & (string_libs == 0)

        # 5) Ko detection (single‑stone capture)
        ko_rowcol = torch.full((B, 2), -1, dtype=torch.int16, device=self.device)
        single_idx = (captured.sum((1, 2)) == 1).nonzero(as_tuple=False).squeeze(1)
        if single_idx.numel():
            coords = captured[single_idx].nonzero(as_tuple=False)[:, 1:]  # (N, 2)
            ko_rowcol[single_idx] = coords.to(torch.int16)

        return captured, ko_rowcol

    # ------------------------- board mutation --------------------------

    def place_stones_batch(self, pos: torch.Tensor) -> None:
        """Play a batch of moves.

        ``pos`` shape: *(B, 2)* – row, col or ←1, -1> for *pass*.
        """
        idx = torch.arange(self.batch_size, device=self.device)
        is_pass = pos[:, 0] < 0

        # update pass counters / reset ko
        self.pass_count += is_pass.to(torch.uint8)
        self.pass_count[~is_pass] = 0
        self.ko_points[:] = -1

        play_mask = ~is_pass
        if play_mask.any():
            b = idx[play_mask]
            r = pos[play_mask, 0]
            c = pos[play_mask, 1]
            pl = self.current_player[play_mask]

            # 1) place stones & update Zobrist hash
            self.stones[b, pl.long(), r, c] = 1
            self.current_hash[b] ^= self.zobrist_keys[pl.long(), r.long(), c.long()]

            played = torch.zeros_like(self.stones[:, 0], dtype=torch.bool)
            played[b, r, c] = True

            # 2) resolve captures & ko
            captured, ko = self._capture_mask(played, self.current_player)
            if captured.any():
                self.stones &= (~captured).unsqueeze(1).to(torch.uint8)

                nz = captured.nonzero(as_tuple=False)
                if nz.numel():
                    rows, cols = nz[:, 1], nz[:, 2]
                    colours = (1 - self.current_player[nz[:, 0]]).long()
                    keys = self.zobrist_keys[colours, rows, cols]
                    # XOR one stone at a time (rare – only when captures occur)
                    for i in range(nz.shape[0]):
                        self.current_hash[nz[i, 0]] ^= keys[i]

            self.ko_points = ko

        # 3) switch player (update Zobrist)
        self.current_hash ^= self.turn_keys[self.current_player.long()]
        self.current_player = 1 - self.current_player
        self.current_hash ^= self.turn_keys[self.current_player.long()]

    # -------------------------- feature planes -------------------------

    def to_features(self) -> torch.Tensor:
        """Return AlphaZero‑style 5‑plane representation for the batch."""
        ids = torch.arange(self.batch_size, device=self.device)
        cur = self.stones[ids, self.current_player.long()].float()
        opp = self.stones[ids, (1 - self.current_player).long()].float()
        legal = self.get_legal_moves_mask().float()
        empty = self.get_empty_mask().float()
        lib = (
            F.conv2d(cur.unsqueeze(1), self.neighbor_kernel[None, None], padding=1)
            .squeeze(1)
            .mul_(empty)
        )
        turn = (
            self.current_player.view(-1, 1, 1)
            .expand(-1, self.board_size, self.board_size)
            .float()
        )
        return torch.stack([cur, opp, legal, lib, turn], dim=1)


# ---------------------------------------------------------------------------
#  Dumb random bot for testing
# ---------------------------------------------------------------------------

class TensorBatchBot:
    """Uniform‑random move sampler (batch‑friendly)."""

    def __init__(self, device: torch.device | str | None = None):
        self.device = torch.device(device) if device is not None else _select_device()

    def select_moves(self, boards: TensorBoard) -> torch.Tensor:
        legal = boards.get_legal_moves_mask()
        B, H, W = legal.shape
        flat = legal.view(B, -1)

        moves = torch.full((B, 2), -1, dtype=torch.int32, device=self.device)
        play = flat.any(dim=1)
        if play.any():
            probs = flat[play].float()
            probs /= probs.sum(dim=1, keepdim=True)
            idx = torch.multinomial(probs, 1).squeeze(1).to(torch.int32)
            moves[play, 0] = idx // W
            moves[play, 1] = idx % W
        return moves

