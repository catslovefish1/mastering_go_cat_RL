# tensor_native.py – Optimised Go engine with GoLegalMoveChecker integration
# (fully int-aligned, no stray .item() calls left)

from __future__ import annotations
import os
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor

# -----------------------------------------------------------------------------#
#  Shared utilities                                                            #
# -----------------------------------------------------------------------------#
from utils.shared import (
    select_device,
    is_pass_move,
    get_batch_indices,
    timed_method,
    print_timing_report,
)

# -----------------------------------------------------------------------------#
#  GoLegalMoveChecker import (robust to package layout)                        #
# -----------------------------------------------------------------------------#
try:                     # 1. relative
    from . import GoLegalMoveChecker as legal_module
except (ImportError, AttributeError):
    try:                 # 2. top-level
        import GoLegalMoveChecker as legal_module
    except (ImportError, AttributeError):  # 3. engine sub-package
        from engine import GoLegalMoveChecker as legal_module

GoLegalMoveChecker = legal_module.GoLegalMoveChecker

# -----------------------------------------------------------------------------#
#  Environment setup                                                           #
# -----------------------------------------------------------------------------#
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ========================= CONSTANTS ========================================= #

@dataclass(frozen=True)
class Stone:
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1


BoardTensor    = Tensor   # (B, H, W)
StoneTensor    = Tensor   # (B, 2, H, W)
PositionTensor = Tensor   # (B, 2)
BatchTensor    = Tensor   # (B,)

# ========================= GO ENGINE ========================================= #

class TensorBoard(torch.nn.Module):
    """Elegant vectorised Go board with batched legal-move checking."""
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
        self.batch_size      = batch_size
        self.board_size      = board_size
        self.history_factor  = history_factor
        self.device          = device or select_device()
        self.enable_timing   = enable_timing

        # Timing helpers
        object.__setattr__(self, "timings",      defaultdict(list) if enable_timing else {})
        object.__setattr__(self, "call_counts",  defaultdict(int)  if enable_timing else {})

        # Core helper
        self.legal_checker = GoLegalMoveChecker(board_size=board_size, device=self.device)

        self._init_zobrist_table()
        self._init_constants()
        self._init_state()
        self._cache = {}

        # Cached legal/capture info (invalidated every move)
        self._last_legal_mask   = None
        self._last_capture_info = None

    # ------------------------------------------------------------------ #
    # Static data                                                        #
    # ------------------------------------------------------------------ #
    def _init_zobrist_table(self) -> None:
        """Stub – fill in if you need Zobrist hashing."""
        pass

    def _init_constants(self) -> None:
        """Neighbour offsets as *plain ints* (N, S, W, E)."""
        self.NEIGHBOR_OFFSETS = (
            -self.board_size,   # North
             self.board_size,   # South
            -1,                 # West
             1,                 # East
        )

    # ------------------------------------------------------------------ #
    # Mutable state                                                      #
    # ------------------------------------------------------------------ #
    def _init_state(self) -> None:
        B, H, W = self.batch_size, self.board_size, self.board_size

        self.register_buffer("stones",         torch.zeros((B, 2, H, W), dtype=torch.bool,  device=self.device))
        self.register_buffer("current_player", torch.zeros(B,               dtype=torch.uint8, device=self.device))
        self.register_buffer("position_hash",  torch.zeros(B,               dtype=torch.int64, device=self.device))
        self.register_buffer("ko_points",      torch.full((B, 2), -1,       dtype=torch.int8,  device=self.device))
        self.register_buffer("pass_count",     torch.zeros(B,               dtype=torch.uint8, device=self.device))

        # Board history: depth = board_size² × history_factor
        max_moves = self.board_size * self.board_size * self.history_factor
        self.register_buffer(
            "board_history",
            torch.full((B, max_moves, H * W), -1, dtype=torch.int8, device=self.device),
        )
        self.register_buffer("move_count", torch.zeros(B, dtype=torch.int16, device=self.device))

    # ==================== CORE UTILITIES ==================================== #
    def switch_player(self) -> None:
        self.current_player = self.current_player ^ 1
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._cache.clear()
        self._last_legal_mask   = None
        self._last_capture_info = None

    # ==================== BOARD QUERIES ===================================== #
    def get_player_stones(self, player: Optional[int] = None) -> BoardTensor:
        if player is None:
            idx = get_batch_indices(self.batch_size, self.device)
            return self.stones[idx, self.current_player.long()]
        return self.stones[:, player]

    def get_opponent_stones(self) -> BoardTensor:
        idx = get_batch_indices(self.batch_size, self.device)
        return self.stones[idx, (1 - self.current_player).long()]

    # ==================== CAPTURE REMOVAL =================================== #
    def _remove_captured_stones_by_root(
        self,
        batch_idx: int,
        root_idx: int,
        roots: torch.Tensor,
        colour: torch.Tensor,
        opponent: int,
    ) -> int:
        """Remove every stone belonging to a captured group; return count."""
        mask           = (roots[batch_idx] == root_idx) & (colour[batch_idx] == opponent)
        captured_count = int(mask.sum())
        if captured_count:
            pos     = mask.nonzero(as_tuple=True)[0]          # flat indices
            rows    = pos // self.board_size
            cols    = pos %  self.board_size
            self.stones[batch_idx, opponent, rows, cols] = False
        return captured_count

    # ==================== KO HANDLING ======================================= #
    def _detect_ko(
        self,
        batch_idx: int,
        row: int,
        col: int,
        captured_count: int,
        capture_positions: List[Tuple[int, int]],
    ) -> None:
        if captured_count == 1 and len(capture_positions) == 1:
            r, c = capture_positions[0]
            self.ko_points[batch_idx] = torch.tensor([r, c], dtype=torch.int8, device=self.device)
        else:
            self.ko_points[batch_idx] = -1

    # ==================== LEGAL MOVES ======================================= #
    @timed_method
    def legal_moves(self) -> BoardTensor:
        legal_mask, capture_info = self.legal_checker.compute_legal_moves_with_captures(
            stones=self.stones,
            current_player=self.current_player,
            ko_points=self.ko_points,
            return_capture_info=True,
        )
        self._last_legal_mask   = legal_mask
        self._last_capture_info = capture_info
        return legal_mask

    @timed_method
    def get_capture_info(self, b: int, r: int, c: int) -> Dict:
        if self._last_capture_info is None:
            self.legal_moves()
        info = self._last_capture_info
        return dict(
            would_capture = info["would_capture"][b, r, c],
            capture_groups= info["capture_groups"][b, r, c],
            capture_sizes = info["capture_sizes"][b, r, c],
            total_captures= info["total_captures"][b, r, c],
        )

    # ==================== MOVE EXECUTION ==================================== #
    def _place_stones(self, positions: PositionTensor) -> None:
        B = positions.size(0)
        for i in range(B):
            row, col = map(int, positions[i].tolist())

            # Pass move?
            if row < 0 or col < 0:
                continue
            if row >= self.board_size or col >= self.board_size:
                print(f"Warning: move {(row, col)} out of bounds");  continue

            player, opponent = int(self.current_player[i]), 1 - int(self.current_player[i])
            self.stones[i, player, row, col] = True

            total_captured   = 0
            capture_positions: list[tuple[int, int]] = []

            if self._last_capture_info is not None:
                info    = self.get_capture_info(i, row, col)
                if info["would_capture"]:
                    roots   = self._last_capture_info["roots"]
                    colour  = self._last_capture_info["colour"]
                    seen_roots: set[int] = set()

                    for dir_idx in range(4):
                        root = int(info["capture_groups"][dir_idx])
                        if root >= 0 and root not in seen_roots:
                            seen_roots.add(root)
                            captured = self._remove_captured_stones_by_root(
                                i, root, roots, colour, opponent
                            )
                            if captured:
                                total_captured += captured
                                flat     = row * self.board_size + col
                                nbr_flat = flat + self.NEIGHBOR_OFFSETS[dir_idx]
                                nbr_row  = nbr_flat // self.board_size
                                nbr_col  = nbr_flat %  self.board_size
                                capture_positions.append((nbr_row, nbr_col))

            if total_captured:
                self._detect_ko(i, row, col, total_captured, capture_positions)

    # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
    def _update_board_history(self) -> None:
        black = self.stones[:, 0].flatten(1)
        white = self.stones[:, 1].flatten(1)
        board_state = torch.where(
            black,
            torch.zeros_like(black, dtype=torch.int8),
            torch.where(
                white,
                torch.ones_like(white, dtype=torch.int8),
                torch.full_like(white, -1, dtype=torch.int8),
            ),
        )
        move_idx = self.move_count.long()
        for b in range(self.batch_size):
            if move_idx[b] < self.board_history.shape[1]:
                self.board_history[b, move_idx[b]] = board_state[b]

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
        self.pass_count = torch.where(is_pass, self.pass_count + 1, torch.zeros_like(self.pass_count))
        self.ko_points[~is_pass] = -1  # clear ko before play moves

        self._place_stones(positions)
        self.switch_player()

    # ------------------------------------------------------------------ #
    # Game state                                                         #
    # ------------------------------------------------------------------ #
    def is_game_over(self) -> BatchTensor:
        return self.pass_count >= 2

    def compute_scores(self) -> Tensor:
        black = self.stones[:, Stone.BLACK].sum((1, 2)).float()
        white = self.stones[:, Stone.WHITE].sum((1, 2)).float()
        return torch.stack([black, white], dim=1)

    # ------------------------------------------------------------------ #
    # Timing                                                             #
    # ------------------------------------------------------------------ #
    def print_timing_report(self, top_n: int = 30) -> None:
        if self.enable_timing:
            print_timing_report(self, top_n)
