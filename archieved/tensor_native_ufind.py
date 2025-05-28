"""tensor_board_optimized_dsu.py – Go engine rebuilt around a Union‑Find (DSU)

Key changes vs. flood‑fill edition
==================================
* Replaces per‑move tensor flood‑fill with **GroupDSU** (union–find + liberty sets)
* No temporary flood tensors → drastically lower CPU memory traffic
* One DSU per colour per board ⇒ still vectorised over a batch (list of dicts)
* Public API and Zobrist hashing unchanged – drop‑in compatible
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

# ───────────────────────────── env ─────────────────────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ──────────────────────────── constants ─────────────────────────
@dataclass(frozen=True)
class Stone:
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1

# type aliases
BoardTensor = Tensor  # (B, H, W)
StoneTensor = Tensor  # (B, 2, H, W)
PositionTensor = Tensor  # (B, 2)
BatchTensor = Tensor  # (B,)

# ─────────────────────── helper decorators ─────────────────────

def with_active_games(method: Callable) -> Callable:
    """Skip finished games inside multi‑board batch."""

    @wraps(method)
    def wrapper(self, *args, active_mask: Optional[BatchTensor] = None, **kwargs):
        if active_mask is None:
            active_mask = ~self.is_game_over()
        if not active_mask.any():
            return None
        return method(self, *args, active_mask=active_mask, **kwargs)

    return wrapper

# ──────────────────────── union‑find DSU ───────────────────────

class GroupDSU:
    """DSU that tracks liberties for every group on one board."""

    def __init__(self, H: int, W: int, device: torch.device):
        self.H, self.W, self.N = H, W, H * W
        self.parent = torch.arange(self.N, device=device, dtype=torch.int32)
        self.size = torch.ones(self.N, device=device, dtype=torch.int16)
        self.libs: list[set[int]] = [set() for _ in range(self.N)]
        self._nbr = (-1, +1, -W, +W)

    # ── classic union–find ──
    def find(self, x: int) -> int:
        p = self.parent[x].item()
        while p != self.parent[p].item():
            self.parent[p] = self.parent[self.parent[p]]  # path compression
            p = self.parent[p].item()
        return p

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.libs[ra].update(self.libs[rb])
        self.libs[rb].clear()
        return ra

    # ── public helpers ──
    def add_stone(self, idx: int, empty_flat: Tensor):
        """Insert a stone at *idx* and merge with direct same‑colour neighbours."""
        root = idx
        self.libs[root].clear()
        for off in self._nbr:
            n = idx + off
            if n < 0 or n >= self.N:
                continue
            if empty_flat[n]:
                self.libs[root].add(n)
            else:  # neighbour stone — expect same colour when called correctly
                root = self.union(root, n)
        # incoming stone removes itself from liberty sets of adjacent enemy groups
        # (caller handles enemy DSU).

    def remove_group(self, root: int, stones_flat: Tensor) -> list[int]:
        """Remove captured *root* group, return list of flat indices removed."""
        captured: list[int] = []
        for i in range(self.N):
            if self.find(i) == root:
                captured.append(i)
                stones_flat[i] = False
                self.parent[i] = i
                self.size[i] = 1
                self.libs[i].clear()
        return captured

# ────────────────────── main engine class ──────────────────────

class TensorBoard(torch.nn.Module):
    """CPU‑friendly Go engine with DSU group tracking."""

    def __init__(self, batch_size: int = 1, board_size: int = 19, device: Optional[torch.device] = None):
        super().__init__()
        self.batch_size, self.board_size = batch_size, board_size
        self.device = device or self.select_device()

        self._init_constants()
        self._init_state()
        self._init_work_buffers()

        # DSU groups – one dict per board
        self._groups = [
            {Stone.BLACK: GroupDSU(board_size, board_size, self.device),
             Stone.WHITE: GroupDSU(board_size, board_size, self.device)}
            for _ in range(batch_size)
        ]

        self._cache: dict[str, Tensor] = {}
        self._cache_valid: dict[str, bool] = {}

    # ───────── device selection ─────────
    @staticmethod
    def select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ───────── static tensors ─────────
    def _init_constants(self):
        torch.manual_seed(42)
        max_hash = torch.iinfo(torch.int64).max
        self.register_buffer(
            "zobrist_stones",
            torch.randint(0, max_hash, (2, self.board_size, self.board_size), dtype=torch.int64, device=self.device),
        )
        self.register_buffer("zobrist_turn", torch.randint(0, max_hash, (2,), dtype=torch.int64, device=self.device))
        self.register_buffer("_batch_range", torch.arange(self.batch_size, device=self.device))

    # ───────── mutable state ─────────
    def _init_state(self):
        B, H, W = self.batch_size, self.board_size, self.board_size
        self.register_buffer("stones", torch.zeros((B, 2, H, W), dtype=torch.bool, device=self.device))
        self.register_buffer("current_player", torch.zeros(B, dtype=torch.uint8, device=self.device))
        self.register_buffer("position_hash", torch.zeros(B, dtype=torch.int64, device=self.device))
        self.register_buffer("ko_points", torch.full((B, 2), -1, dtype=torch.int16, device=self.device))
        self.register_buffer("pass_count", torch.zeros(B, dtype=torch.uint8, device=self.device))

    # ───────── scratch buffers ─────────
    def _init_work_buffers(self):
        B, H, W = self.batch_size, self.board_size, self.board_size
        self.register_buffer("_neighbor_work", torch.zeros((B, H, W), dtype=torch.float32, device=self.device))
        self.register_buffer("_move_mask", torch.zeros((B, H, W), dtype=torch.bool, device=self.device))

    # ─────────────────── utilities ────────────────────
    def _count_neighbors(self, mask: BoardTensor) -> BoardTensor:
        nw = self._neighbor_work
        nw.zero_()
        nw[:, 1:, :] += mask[:, :-1, :].float()
        nw[:, :-1, :] += mask[:, 1:, :].float()
        nw[:, :, 1:] += mask[:, :, :-1].float()
        nw[:, :, :-1] += mask[:, :, 1:].float()
        return nw

    def _invalidate_cache(self, full: bool = False):
        if full:
            self._cache.clear(); self._cache_valid.clear()
        else:
            self._cache_valid["empty"] = False
            self._cache_valid["legal"] = False

    # ───────── board queries (cached) ─────────
    @property
    def empty_mask(self) -> BoardTensor:
        if not self._cache_valid.get("empty", False):
            self._cache["empty"] = ~self.stones.any(dim=1)
            self._cache_valid["empty"] = True
        return self._cache["empty"]

    def get_player_stones(self, p: Optional[int] = None) -> BoardTensor:
        if p is None:
            return self.stones[self._batch_range, self.current_player.long()]
        return self.stones[:, p]

    def get_opponent_stones(self) -> BoardTensor:
        opp = 1 - self.current_player
        return self.stones[self._batch_range, opp.long()]

    # ───────── legal move generator ─────────
    def _find_capture_moves(self) -> Optional[BoardTensor]:
        opp = self.get_opponent_stones()
        if not opp.any():
            return None
        empty = self.empty_mask
        liberties = self._count_neighbors(empty)
        vulnerable = opp & (liberties == 1)
        if not vulnerable.any():
            return None
        return empty & (self._count_neighbors(vulnerable) > 0)

    def _apply_ko_restrictions(self, legal: BoardTensor) -> BoardTensor:
        has_ko = self.ko_points[:, 0] >= 0
        if not has_ko.any():
            return legal
        ko_batch = has_ko.nonzero(as_tuple=True)[0]
        legal = legal.clone()
        legal[ko_batch, self.ko_points[ko_batch, 0], self.ko_points[ko_batch, 1]] = False
        return legal

    def legal_moves(self) -> BoardTensor:
        if self._cache_valid.get("legal", False):
            return self._cache["legal"]
        finished = self.is_game_over()
        empty = self.empty_mask
        legal = empty & (self._count_neighbors(empty) > 0)
        cap = self._find_capture_moves()
        if cap is not None:
            legal |= cap
        legal = self._apply_ko_restrictions(legal)
        legal &= ~finished.view(-1, 1, 1)
        self._cache["legal"], self._cache_valid["legal"] = legal, True
        return legal

    # ───────── turn toggling + hashing ─────────
    def _update_hash(self, pos: Tuple[Tensor, Tensor, Tensor], colours: Tensor):
        b, r, c = pos
        if b.numel() == 0:
            return
        flat = r * self.board_size + c
        hv = self.zobrist_stones[colours, r, c]
        self.position_hash[b] ^= hv

    @with_active_games
    def _toggle_turn(self, *, active_mask: BatchTensor):
        self.position_hash[active_mask] ^= self.zobrist_turn[self.current_player[active_mask].long()]
        self.current_player[active_mask] ^= 1
        self.position_hash[active_mask] ^= self.zobrist_turn[self.current_player[active_mask].long()]

    # ───────── game logic ─────────
    def step(self, positions: PositionTensor):
        """Apply a vector of moves (row, col) or (-1, -1) for pass."""
        self._invalidate_cache()
        is_pass = positions[:, 0] < 0
        finished = self.is_game_over()
        playing = ~is_pass & ~finished

        # pass handling
        self.pass_count = torch.where(is_pass | finished, self.pass_count + 1, torch.zeros_like(self.pass_count))
        self.ko_points[~finished] = -1

        if playing.any():
            idx = playing.nonzero(as_tuple=True)[0]
            self._play_stones(idx, positions[playing])
        self._toggle_turn()

    def _play_stones(self, batch_idx: Tensor, pos: PositionTensor):
        rows, cols = pos[:, 0].long(), pos[:, 1].long()
        colours = self.current_player[batch_idx].long()
        bs = self.board_size

        for i in range(batch_idx.size(0)):
            b = batch_idx[i].item()
            r, c, col = rows[i].item(), cols[i].item(), colours[i].item()
            flat = r * bs + c

            # place stone
            self.stones[b, col, r, c] = True
            self._update_hash((batch_idx.new_tensor([b]), rows[i:i+1], cols[i:i+1]), colours[i:i+1])

            # DSU update for own colour
            empty_flat = (~self.stones[b].any(dim=0)).view(-1)
            self._groups[b][col].add_stone(flat, empty_flat)

            # capture detection on opponent groups touching the stone
            opp = 1 - col
            captured_b, captured_r, captured_c = [], [], []
            for off in (-1, +1, -bs, +bs):
                n = flat + off
                if n < 0 or n >= bs * bs:
                    continue
                nr, nc = divmod(n, bs)
                if not self.stones[b, opp, nr, nc]:
                    continue
                root = self._groups[b][opp].find(n)
                if len(self._groups[b][opp].libs[root]) == 0:
                    cap = self._groups[b][opp].remove_group(root, self.stones[b, opp].view(-1))
                    for p in cap:
                        pr, pc = divmod(p, bs)
                        captured_b.append(b)
                        captured_r.append(pr)
                        captured_c.append(pc)
            if captured_b:
                cb = torch.tensor(captured_b, device=self.device)
                cr = torch.tensor(captured_r, device=self.device)
                cc = torch.tensor(captured_c, device=self.device)
                self._update_hash((cb, cr, cc), torch.full_like(cb, opp))
                # KO detection (single capture)
                self._detect_ko((cb, cr, cc))

    # ───────── ko detection (unchanged) ─────────
    def _detect_ko(self, captured: Tuple[Tensor, Tensor, Tensor]):
        b, r, c = captured
        if b.numel() == 0:
            return
        cnt = torch.bincount(b, minlength=self.batch_size)
        single = cnt == 1
        if not single.any():
            return
        batch_to_idx = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        batch_to_idx.scatter_(0, b, torch.arange(b.numel(), device=self.device))
        sel = single.nonzero(as_tuple=True)[0]
        valid_idx = batch_to_idx[sel]
        self.ko_points[sel, 0] = r[valid_idx].to(torch.int16)
        self.ko_points[sel, 1] = c[valid_idx].to(torch.int16)

    # ───────── scoring / termination ─────────
    def is_game_over(self) -> BatchTensor:
        return self.pass_count >= 2

    def compute_scores(self) -> Tensor:
        black = self.stones[:, Stone.BLACK].sum(dim=(1, 2)).float()
        white = self.stones[:, Stone.WHITE].sum(dim=(1, 2)).float()
        return torch.stack([black, white], dim=1)

    # ───────── feature extraction ─────────
    def extract_features(self) -> Tensor:
        cur = self.get_player_stones().float()
        opp = self.get_opponent_stones().float()
        legal = self.legal_moves().float()
        empty = self.empty_mask
        cur_libs = self._count_neighbors(empty) * cur
        turn = self.current_player.float().view(-1, 1, 1).expand(-1, self.board_size, self.board_size)
        return torch.stack([cur, opp, legal, cur_libs, turn], dim=1)

    # ───────── utility ─────────
    def to_numpy(self, idx: int = 0):
        import numpy as np
        board = np.full((self.board_size, self.board_size), Stone.EMPTY, dtype=np.int8)
        board[self.stones[idx, Stone.BLACK].cpu().numpy()] = Stone.BLACK
        board[self.stones[idx, Stone.WHITE].cpu().numpy()] = Stone.WHITE
        return board
