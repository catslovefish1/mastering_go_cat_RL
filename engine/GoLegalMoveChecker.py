# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py – **board-plane edition** (2025-06-04)
===========================================================

This is a *drop-in* replacement for the previous “v2-batched” file, updated to
operate directly on a single **int8 board tensor** shaped **(B, H, W)** whose
cell values are

    -1  empty
     0  black
     1  white

All public APIs and return values are unchanged except that the `stones`
argument has been renamed to `board` (and now expects the new layout).  No
helper conversions are required anywhere in the engine stack.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Union

import torch

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
DTYPE:     torch.dtype = torch.int16
IDX_DTYPE: torch.dtype = torch.int64


# =============================================================================
# Public API
# =============================================================================
class GoLegalMoveChecker:
    """Vectorised legal-move checker with capture detection for Go."""

    def __init__(
        self,
        board_size: int = 19,
        device: Optional[torch.device] = None,
    ):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device

        # Internal fully-batched checker (now board-native as well)
        self._checker = VectorizedBoardChecker(board_size, self.device)

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def compute_legal_moves_with_captures(
        self,
        board: torch.Tensor,               # (B, H, W)  int8   –-1/0/1
        current_player: torch.Tensor,      # (B,)        uint8  0/1
        ko_points: Optional[torch.Tensor] = None,  # (B, 2) int8  row, col
        return_capture_info: bool = True,
    ) -> Union[torch.Tensor,
               Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the legal-move mask for each board in the batch.

        Returns
        -------
        legal_mask : (B, H, W) bool
        capture_info : dict of tensors (only if *return_capture_info* is True)
        """
        B, H, W = board.shape
        assert H == self.board_size and W == self.board_size, "board size mismatch"

        # Heavy lifting
        legal_mask, capture_info = self._checker.compute_batch_legal_and_captures(
            board, current_player
        )

        # Simple Ko-rule masking (cheap)
        if ko_points is not None:
            ko_valid = ko_points[:, 0] >= 0
            if ko_valid.any():
                b = ko_valid.nonzero(as_tuple=True)[0]
                r = ko_points[b, 0].long()
                c = ko_points[b, 1].long()
                legal_mask[b, r, c] = False

        return (legal_mask, capture_info) if return_capture_info else legal_mask


# =============================================================================
# Batch-vectorised board checker
# =============================================================================
class VectorizedBoardChecker:
    """
    Fully batched legal-move logic with capture detection.
    Works directly on *(B, H, W)* int8 boards (-1/0/1).
    """

    # --------------------------------------------------------------
    # Construction helpers
    # --------------------------------------------------------------
    def __init__(self, board_size: int, device: torch.device | None):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device
        self._init_neighbor_structure()

    def _init_neighbor_structure(self) -> None:
        """Pre-compute neighbour indices & validity mask (CPU-free during play)."""
        N = self.board_size
        OFF   = torch.tensor([-N, N, -1, 1], dtype=IDX_DTYPE, device=self.device)
        flat  = torch.arange(self.N2, dtype=IDX_DTYPE, device=self.device)
        nbrs  = flat[:, None] + OFF                      # (N², 4)

        valid = (nbrs >= 0) & (nbrs < self.N2)
        col   = flat % N
        valid[:, 2] &= col != 0          # West edge
        valid[:, 3] &= col != N - 1      # East edge

        self.NEIGH_IDX   = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.NEIGH_VALID = valid

    # --------------------------------------------------------------
    # Top-level batched computation
    # --------------------------------------------------------------
    def compute_batch_legal_and_captures(
        self,
        board: torch.Tensor,          # (B, H, W)  int8  –-1/0/1
        current_player: torch.Tensor  # (B,)        uint8 0/1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, H, W = board.shape
        N2 = self.N2

        board_f  = board.view(B, N2)                 # (B, N²) int8
        occupied = board_f != -1                     # bool
        empty    = ~occupied

        # Union-find on occupied stones
        _, colour, roots, root_libs = self._batch_init_union_find(board_f)

        curr_player = current_player.view(B, 1)         # (B,1)
        opp_player  = 1 - curr_player

        # Neighbour look-ups
        neigh_colors = self._get_neighbor_colors_batch(colour)   # (B,N²,4)
        neigh_roots  = self._get_neighbor_roots_batch(roots)     # (B,N²,4)
        valid_mask   = self.NEIGH_VALID.view(1, N2, 4)

        # 1-lib moves (simple liberties)
        has_any_lib = ((neigh_colors == -1) & valid_mask).any(dim=2)

        # Capture moves (neighbour opponent group with exactly 1 liberty)
        opp_mask       = (neigh_colors == opp_player.view(B, 1, 1)) & valid_mask
        neigh_roots_f  = neigh_roots.reshape(B, -1)
        neigh_libs_f   = root_libs.gather(1, neigh_roots_f.clamp(min=0))
        neigh_libs     = neigh_libs_f.view(B, N2, 4)
        can_capture    = opp_mask & (neigh_libs == 1)
        can_capture_any = can_capture.any(dim=2)

        # Friendly extensions (connect to own group that has >1 liberty)
        friendly       = (neigh_colors == curr_player.view(B, 1, 1)) & valid_mask
        friendly_libs  = friendly & (neigh_libs > 1)
        friendly_any   = friendly_libs.any(dim=2)

        # Final legality: empty AND (liberty OR capture OR friendly-extension)
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)

        # ------------------- Capture meta-data -------------------
        capture_groups = torch.full((B, N2, 4),
                                    -1, dtype=IDX_DTYPE, device=self.device)
        capture_groups[can_capture] = neigh_roots[can_capture]

        sizes = torch.zeros((B, N2), dtype=IDX_DTYPE, device=self.device)
        sizes.scatter_add_(1, roots, torch.ones_like(roots, dtype=IDX_DTYPE))

        capture_sizes = torch.zeros_like(capture_groups)
        valid_cap     = capture_groups >= 0
        cap_flat      = capture_groups.view(B, -1)
        valid_flat    = valid_cap.view(B, -1)
        sizes_flat    = sizes.gather(1, cap_flat.clamp(min=0))
        capture_sizes.view(B, -1)[valid_flat] = sizes_flat[valid_flat]

        total_captures = capture_sizes.sum(dim=2).view(B, H, W)

        capture_info: Dict[str, torch.Tensor] = {
            "would_capture" : (empty & can_capture_any).view(B, H, W),
            "capture_groups": capture_groups.view(B, H, W, 4),
            "capture_sizes" : capture_sizes.view(B, H, W, 4),
            "total_captures": total_captures,
            "roots"         : roots,
            "colour"        : colour,
        }
        return legal_mask, capture_info

    # --------------------------------------------------------------
    # Tensorised neighbour helpers
    # --------------------------------------------------------------
    def _get_neighbor_colors_batch(self, colour: torch.Tensor) -> torch.Tensor:
        B, N2 = colour.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)

        idx_flat   = idx.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)

        neigh_flat = torch.full_like(idx_flat, -2, dtype=DTYPE)
        gathered   = torch.gather(colour, 1, idx_flat.clamp(min=0))
        neigh_flat[valid_flat] = gathered[valid_flat]
        return neigh_flat.view(B, N2, 4)

    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        B, N2 = roots.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)

        idx_flat   = idx.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)

        neigh_flat = torch.full_like(idx_flat, -1, dtype=IDX_DTYPE)
        gathered   = torch.gather(roots, 1, idx_flat.clamp(min=0))
        neigh_flat[valid_flat] = gathered[valid_flat]
        return neigh_flat.view(B, N2, 4)

    # --------------------------------------------------------------
    # Fully batched union-find initialisation (no Python loops)
    # --------------------------------------------------------------
    def _batch_init_union_find(
        self,
        board_f: torch.Tensor  # (B, N²) int8
    ) -> Tuple[torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        board_f : (B, N²) int8  – values -1/0/1

        Returns
        -------
        parent      : (B, N²) int64
        colour      : (B, N²) int16  (-1/0/1)
        roots       : (B, N²) int64  – root index of each stone
        root_libs   : (B, N²) int64  – liberties count per root id
        """
        B, N2 = board_f.shape
        dev = self.device

        # Colour map (-1 empty, 0 black, 1 white)
        colour = board_f.to(DTYPE)

        # Initialise each occupied point as its own parent
        parent = torch.arange(N2, dtype=IDX_DTYPE, device=dev).repeat(B, 1)

        # ---------- merge neighbouring stones of the same colour ----------
        neigh_cols = self._get_neighbor_colors_batch(colour)                 # (B,N²,4)
        same       = (neigh_cols == colour.unsqueeze(2)) \
                     & (colour.unsqueeze(2) != -1) \
                     & self.NEIGH_VALID.view(1, N2, 4)

        if same.any():
            # Flatten indices for batched union
            b, p, d = same.nonzero(as_tuple=True)                 # lists
            nbr     = self.NEIGH_IDX[p, d]                        # flat neighbour index
            gpos    = b * N2 + p                                  # global indices
            gnbr    = b * N2 + nbr

            parent_flat = parent.view(-1)
            new_par     = torch.minimum(parent_flat[gpos], parent_flat[gnbr])
            parent_flat[gpos] = new_par
            parent_flat[gnbr] = new_par

            # A few halving passes for compression
            for _ in range(5):
                parent_flat[:] = parent_flat[parent_flat]

            parent = parent_flat.view(B, N2)

        roots = parent.clone()   # after compression

        # ---------- count unique liberties per root ----------
        neigh_cols = self._get_neighbor_colors_batch(colour)
        is_lib     = (neigh_cols == -1) & self.NEIGH_VALID.view(1, N2, 4)
        stone_mask = colour != -1

        libs_per_root = torch.zeros(B * N2, dtype=IDX_DTYPE, device=dev)

        if stone_mask.any():
            batch_map = torch.arange(B, dtype=IDX_DTYPE, device=dev).view(B, 1, 1)
            roots_exp = roots.unsqueeze(2)            # (B,N²,1)
            lib_idx   = self.NEIGH_IDX.view(1, N2, 4) # broadcast (1,N²,4)
            mask      = is_lib & stone_mask.unsqueeze(2)

            fb = batch_map.expand_as(mask)[mask]      # (K,)
            fr = roots_exp.expand_as(mask)[mask]      # (K,)
            fl = lib_idx.expand_as(mask)[mask]        # (K,)

            key_root = fb * N2 + fr       # unique per game+root
            key_lib  = fb * N2 + fl       # unique per game+point
            pairs    = torch.stack((key_root, key_lib), 1)

            # sort by lib to enable unique_consecutive
            pairs = pairs[pairs[:, 1].argsort()]
            uniq  = torch.unique_consecutive(pairs, dim=0)

            libs_per_root.scatter_add_(0,
                                        uniq[:, 0],
                                        torch.ones_like(uniq[:, 0], dtype=IDX_DTYPE))

        root_libs = libs_per_root.view(B, N2)
        return parent, colour, roots, root_libs
