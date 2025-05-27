from __future__ import annotations
"""tensor_board_optimized.py – fully vectorised Go engine (GPU‑friendly).

*Buffers now live on `self.device` from the first allocation* to avoid the
CPU/GPU mismatch that caused the `RuntimeError: indices should be either on cpu
or on the same device` during `legal_moves`.
"""

import os
from typing import Tuple

import torch
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


BLACK, WHITE, EMPTY = 0, 1, -1


class TensorBoard(torch.nn.Module):
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.device = torch.device(device) if device is not None else _select_device()
        self.batch_size = int(batch_size)
        self.board_size = int(board_size)

        # CONSTANT BUFFERS --------------------------------------------------
        neighbour = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=self.device)
        self.register_buffer("neighbor_kernel", neighbour[None, None])  # (1,1,3,3)

        torch.manual_seed(42)
        max64 = torch.iinfo(torch.int64).max
        self.register_buffer(
            "zobrist_keys",
            torch.randint(0, max64, (2, board_size, board_size), dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "turn_keys",
            torch.randint(0, max64, (2,), dtype=torch.int64, device=self.device),
        )

        # MUTABLE STATE BUFFERS --------------------------------------------
        self.register_buffer(
            "stones",
            torch.zeros((batch_size, 2, board_size, board_size), dtype=torch.bool, device=self.device),
        )
        self.register_buffer("current_player", torch.zeros(batch_size, dtype=torch.uint8, device=self.device))
        self.register_buffer("current_hash", torch.zeros(batch_size, dtype=torch.int64, device=self.device))
        self.register_buffer("ko_points", torch.full((batch_size, 2), -1, dtype=torch.int16, device=self.device))
        self.register_buffer("pass_count", torch.zeros(batch_size, dtype=torch.uint8, device=self.device))

    # ----------------------------- HELPERS --------------------------------
    def _empty_mask(self) -> torch.Tensor:
        return ~(self.stones[:, BLACK] | self.stones[:, WHITE])

    # -------------------------- LEGAL MOVES --------------------------------
    def legal_moves(self) -> torch.Tensor:
        B = self.batch_size
        empty = self._empty_mask()

        lib_cnt = F.conv2d(empty.float().unsqueeze(1), self.neighbor_kernel, padding=1).squeeze(1)
        legal = empty & (lib_cnt > 0)

        ids = torch.arange(B, device=self.device)
        opp = self.stones[ids, (1 - self.current_player).long()]
        opp_lib = F.conv2d(opp.float().unsqueeze(1), self.neighbor_kernel, padding=1).squeeze(1)
        weak = opp & (opp_lib == 1)
        adj_weak = F.conv2d(weak.float().unsqueeze(1), self.neighbor_kernel, padding=1).squeeze(1) > 0
        legal |= empty & adj_weak

        has_ko = self.ko_points[:, 0] >= 0
        if has_ko.any():
            rows = self.ko_points[has_ko, 0].long()
            cols = self.ko_points[has_ko, 1].long()
            legal[has_ko, rows, cols] = False

        return legal

    # ------------------------------ STEP ----------------------------------
    def step(self, pos: torch.Tensor) -> None:
        if pos.shape != (self.batch_size, 2):
            raise ValueError("pos must have shape (batch_size,2)")

        B = self.batch_size
        ids = torch.arange(B, device=self.device)

        is_pass = pos[:, 0] < 0
        self.pass_count += is_pass.to(torch.uint8)
        self.pass_count[~is_pass] = 0
        self.ko_points.fill_(-1)

        self.current_hash ^= self.turn_keys[self.current_player.long()]

        play = ~is_pass
        if play.any():
            r = pos[play, 0].long()
            c = pos[play, 1].long()
            who = self.current_player[play].long()

            self.stones[ids[play], who, r, c] = True

            flat = r * self.board_size + c
            delta = self.zobrist_keys[who].flatten()[flat]
            upd = torch.zeros_like(self.current_hash)
            upd[play] = delta
            self.current_hash ^= upd

            self._captures(ids[play], r, c, who)

        self.current_player ^= 1
        self.current_hash ^= self.turn_keys[self.current_player.long()]

    # ---------------------------- CAPTURES --------------------------------
    def _captures(self, b_idx: torch.Tensor, r: torch.Tensor, c: torch.Tensor, col: torch.Tensor) -> None:
        if b_idx.numel() == 0:
            return
        H = self.board_size
        nbr = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]], device=self.device)

        coords = torch.stack([r, c], 1).unsqueeze(1) + nbr
        coords = coords.view(-1, 2)
        on = (coords[:, 0] >= 0) & (coords[:, 0] < H) & (coords[:, 1] >= 0) & (coords[:, 1] < H)
        coords = coords[on]
        b_for = b_idx.repeat_interleave(4)[on]
        opp_col = (1 - col.repeat_interleave(4))[on]

        cand = torch.zeros_like(self.stones[:, 0])
        cand[b_for, coords[:, 0].long(), coords[:, 1].long()] = True
        opp_plane = self.stones[torch.arange(self.batch_size, device=self.device), opp_col]
        cand &= opp_plane
        if not cand.any():
            return

        grown = cand.clone()
        while True:
            dil = F.conv2d(grown.float().unsqueeze(1), self.neighbor_kernel, padding=1).squeeze(1) > 0
            new = dil & opp_plane
            if not (new & ~grown).any():
                break
            grown |= new

        empty = self._empty_mask()
        libs = F.conv2d(grown.float().unsqueeze(1), self.neighbor_kernel, padding=1).squeeze(1) & empty
        captured = grown & ~libs.any(dim=0, keepdim=True)
        if not captured.any():
            return

        cap = captured.nonzero(as_tuple=False)
        b_cap, r_cap, c_cap = cap[:, 0], cap[:, 1], cap[:, 2]
        opp = (1 - self.current_player[b_cap]).long()

        self.stones[b_cap, opp, r_cap, c_cap] = False
        self.current_hash[b_cap] ^= self.zobrist_keys[opp, r_cap, c_cap]

        for b in b_cap.unique():
            mask = b_cap == b
            if mask.sum() == 1:
                self.ko_points[b] = torch.stack([r_cap[mask], c_cap[mask]]).to(torch.int16)

    # ------------------------- TERMINATION & SCORE ------------------------
    def is_game_over(self) -> torch.Tensor:
        return self.pass_count >= 2

    def score(self) -> torch.Tensor:
        b = self.stones[:, BLACK].sum((1, 2)).float()
        w = self.stones[:, WHITE].sum((1, 2)).float()
        return torch.stack([b, w], 1)

    # --------------------------- FEATURES ---------------------------------
    def features(self) -> torch.Tensor:
        ids = torch.arange(self.batch_size, device=self.device)
        cur = self.stones[ids, self.current_player.long()].float()
        opp = self.stones[ids, (1 - self.current_player).long()].float()
        legal = self.legal_moves().float()
        empty = self._empty_mask().float()
        cur_lib = F.conv2d(cur.unsqueeze(1), self.neighbor_kernel, padding=1).squeeze(1) * empty
        turn = self.current_player.view(-1, 1, 1).expand(-1, self.board_size, self.board_size).float()
        return torch.stack([cur, opp, legal, cur_lib, turn], 1)

    # ----------------------------- DEBUG ----------------------------------
    def numpy_board(self, idx: int = 0):
        import numpy as np

        board = np.full((self.board_size, self.board_size), -1, dtype=np.int8)
        board[self.stones[idx, BLACK].cpu().numpy()] = 0
        board[self.stones[idx, WHITE].cpu().numpy()] = 1
        return board


try:
    TensorBoard = torch.compile(TensorBoard, dynamic=True, fullgraph=False)  # type: ignore
except Exception:
    pass
