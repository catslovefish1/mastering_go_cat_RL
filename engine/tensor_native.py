# tensor_native.py – optimised Go engine with GoLegalMoveChecker integration
# fully int‑aligned, hot‑path free of .item() / .tolist(), and 100 % batched capture/ko bookkeeping.

from __future__ import annotations
import os
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch import Tensor

# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------
from utils.shared import (
    select_device,
    timed_method,
    print_timing_report,
)

# -----------------------------------------------------------------------------
# GoLegalMoveChecker import
# -----------------------------------------------------------------------------

from engine import GoLegalMoveChecker as legal_module
GoLegalMoveChecker = legal_module.GoLegalMoveChecker

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ========================= CONSTANTS =========================================

@dataclass(frozen=True)
class Stone:
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1


BoardTensor    = Tensor   # (B, H, W)
PositionTensor = Tensor   # (B, 2)
PassTensor    = Tensor   # (B,)

# ========================= GO ENGINE =========================================

class TensorBoard(torch.nn.Module):
    """Vectorised multi‑game Go board with batched legal‑move checking."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        history_factor: int = 10,
        device: Optional[torch.device] = None,
        enable_timing: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size     = batch_size
        self.board_size     = board_size
        self.history_factor = history_factor
        self.device         = device or select_device()
        self.enable_timing  = enable_timing

        object.__setattr__(self, "timings",
                           defaultdict(list) if enable_timing else {})
        object.__setattr__(self, "call_counts",
                           defaultdict(int)  if enable_timing else {})

        # Core helper
        self.legal_checker = GoLegalMoveChecker(board_size=board_size,
                                                device=self.device)

        self._init_zobrist_table()
        self._init_constants()
        self._init_state()

        self._cache = {}
        self._last_legal_mask   = None
        self._last_capture_info = None

    # ------------------------------------------------------------------ #
    # Static data                                                        #
    # ------------------------------------------------------------------ #
    def _init_zobrist_table(self) -> None:
        pass  # left as stub

    def _init_constants(self) -> None:
        self.NEIGHBOR_OFFSETS = (
            -self.board_size,   # N
             self.board_size,   # S
            -1,                 # W
             1,                 # E
        )

    # ------------------------------------------------------------------ #
    # Mutable state                                                      #
    # ------------------------------------------------------------------ #
    def _init_state(self) -> None:
        B, H, W = self.batch_size, self.board_size, self.board_size
        dev     = self.device

        self.register_buffer("board",
                     torch.full((B, H, W), Stone.EMPTY,
                                dtype=torch.int8, device=dev))
        self.register_buffer("current_player",
                             torch.zeros(B, dtype=torch.uint8, device=dev))
        self.register_buffer("ko_points",
                             torch.full((B, 2), -1, dtype=torch.int8,
                                        device=dev))
        self.register_buffer("pass_count",
                             torch.zeros(B, dtype=torch.uint8, device=dev))

        max_moves = self.board_size * self.board_size * self.history_factor
        self.register_buffer(
            "board_history",
            torch.full((B, max_moves, H * W), -1, dtype=torch.int8, device=dev)
        )
        self.register_buffer("move_count",
                             torch.zeros(B, dtype=torch.int16, device=dev))

    # ==================== CORE UTILITIES ==================================== #
    def switch_player(self) -> None:
        self.current_player = self.current_player ^ 1
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._cache.clear()
        self._last_legal_mask   = None
        self._last_capture_info = None

    # ==================== CAPTURE / KO HELPERS ============================== #
    def _remove_captured_stones_by_root(
        self,
        batch_idx: int,
        root_idx: int,
        roots: torch.Tensor,
        colour: torch.Tensor,
        opponent: int,
    ) -> int:
        mask = (roots[batch_idx] == root_idx) & (colour[batch_idx] == opponent)
        ncap = int(mask.sum())
        if ncap:
            pos  = mask.nonzero(as_tuple=True)[0]
            rows = pos // self.board_size
            cols = pos %  self.board_size
            self.stones[batch_idx, opponent, rows, cols] = False
        return ncap

    # ------------------------------------------------------------------ #
    # Legal moves and capture info wrappers                              #
    # ------------------------------------------------------------------ #
    @timed_method
    def legal_moves(self) -> BoardTensor:
        legal_mask, cap_info = self.legal_checker.compute_legal_moves_with_captures(
            board=self.board,
            current_player=self.current_player,
            ko_points=self.ko_points,
            return_capture_info=True,
        )
        self._last_legal_mask   = legal_mask
        self._last_capture_info = cap_info
        return legal_mask

    # ==================== MOVE EXECUTION ==================================== #

    @timed_method
    def _place_stones(self, positions: PositionTensor) -> None:
        """Vectorised stone placement **and** capture / ko bookkeeping."""
        H = W = self.board_size

        # ------------------------------------------------------------------ #
        # 1) Vectorised stone placement                                     #
        # ------------------------------------------------------------------ #

        mask_play = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)
        if mask_play.any():
            
            # print(positions.device) 
            b_idx = mask_play.nonzero(as_tuple=True)[0]
            rows  = positions[b_idx, 0].long()
            cols  = positions[b_idx, 1].long()
            ply   = self.current_player[b_idx].long()
            self.board[b_idx, rows, cols] = ply.to(self.board.dtype)   # int8 ← int8
  
        else:
            b_idx = None  # no real moves this turn

        # ------------------------------------------------------------------ #
        # 2) Build capture mask in pure tensor code                         #
        # ------------------------------------------------------------------ #
        roots      = self._last_capture_info["roots"]      # (B, N²)
        colour     = self._last_capture_info["colour"]     # (B, N²)
        cap_groups = self._last_capture_info["capture_groups"]  # (B,H,W,4)
        cap_sizes  = self._last_capture_info["capture_sizes"]   # (B,H,W,4)
        total_caps = self._last_capture_info["total_captures"]  # (B,H,W)

        rows  = positions[b_idx, 0].long()
        cols  = positions[b_idx, 1].long()
        flat  = rows * W + cols
        opp   = (1 - self.current_player[b_idx]).long()

        neigh_roots = cap_groups[b_idx, rows, cols]            # (M,4)
        valid_root  = neigh_roots >= 0

        roots_sel  = roots[b_idx]                               # (M,N²)
        colour_sel = colour[b_idx]

        eq_root = (roots_sel.unsqueeze(2) == neigh_roots.unsqueeze(1)) & valid_root.unsqueeze(1)
        cap_mask_flat = eq_root.any(dim=2) & (colour_sel == opp.unsqueeze(1))
        cap_mask = cap_mask_flat.view(-1, H, W)                 # (M,H,W)

        # ------------------------------------------------------------------ #
        # 3) Clear captured stones & update ko                            #
        # ------------------------------------------------------------------ #
        self.board[b_idx] = torch.where(cap_mask,
                                Stone.EMPTY,
                                self.board[b_idx])

        single_cap = (total_caps[b_idx, rows, cols] == 1)
        if single_cap.any():
            sizes_here = cap_sizes[b_idx, rows, cols]
            dir_single = (sizes_here == 1).float().argmax(dim=1)
            nbr_flat   = flat + torch.tensor(self.NEIGHBOR_OFFSETS, device=self.device)[dir_single]
            r_ko = nbr_flat // W          # row index of the ko-point
            c_ko = nbr_flat %  W          # column index

            self.ko_points[b_idx[single_cap]] = (
            torch.stack([r_ko, c_ko], dim=1)
            .to(self.ko_points.dtype)          #  ← added cast to int8
            )[single_cap]


        # clear ko on boards that didn't make a single‑stone capture
        self.ko_points[b_idx[~single_cap]] = -1



    # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
        # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
    def _update_board_history(self) -> None:
        """
        Record the current position for every live game in the batch.

        board_history  : (B, max_moves, H*W)  int8   – -1 empty, 0 black, 1 white
        move_count[b]  : how many moves have already been written for board b
        """
        B, H, W = self.batch_size, self.board_size, self.board_size
        max_moves = self.board_history.shape[1]

        # ------------------------------------------------------------------
        # 1) Flatten current stones into a single int8 board_state
        # ------------------------------------------------------------------
        flat   = self.board.flatten(1)            # int8
        black  = flat == Stone.BLACK
        white  = flat == Stone.WHITE

        board_state = torch.full_like(black, -1, dtype=torch.int8)   # start as empty
        board_state[black]                   = 0                    # black stones
        board_state[(~black) & white]        = 1                    # white stones

        # ------------------------------------------------------------------
        # 2) Batch-wise write into history (no Python loop)
        # ------------------------------------------------------------------
        move_idx = self.move_count.long()                             # [B]
        valid    = move_idx < max_moves                               # boards not past limit
        if valid.any():
            b_idx   = torch.arange(B, device=self.device)[valid]      # [M]
            mv_idx  = move_idx[valid]                                 # [M]
            self.board_history[b_idx, mv_idx] = board_state[b_idx]    # vectorised write


    # ------------------------------------------------------------------ #
    # Game loop                                                          #
    # ------------------------------------------------------------------ #
    @timed_method
    def step(self, positions: PositionTensor) -> None:
        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError("positions must be (B, 2)")
        if positions.size(0) != self.batch_size:
            raise ValueError("batch size mismatch")

        self._update_board_history()
        self.move_count += 1

        is_pass = (positions[:, 0] < 0) | (positions[:, 1] < 0)
        self.pass_count = torch.where(
            is_pass,
            self.pass_count + 1,
            torch.zeros_like(self.pass_count),
        )
        self.ko_points[~is_pass] = -1  # clear ko before play moves

        self._place_stones(positions)
        self.switch_player()

    # ------------------------------------------------------------------ #
    # Game state                                                         #
    # ------------------------------------------------------------------ #
    def is_game_over(self) -> PassTensor:
        return self.pass_count >= 2

    def compute_scores(self) -> Tensor:
        black = (self.board == Stone.BLACK).sum((1, 2)).float()
        white = (self.board == Stone.WHITE).sum((1, 2)).float()
        return torch.stack([black, white], dim=1)

    # ------------------------------------------------------------------ #
    # Timing                                                             #
    # ------------------------------------------------------------------ #
    def print_timing_report(self, top_n: int = 30) -> None:
        if self.enable_timing:
            print_timing_report(self, top_n)
