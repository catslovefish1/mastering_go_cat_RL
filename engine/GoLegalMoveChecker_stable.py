# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py (v2-batched) – fully-vectorised legal-move computation for Go
=================================================================================

This revision removes the last Python-level loop over the batch dimension by
vectorising the union-find initialisation.  The public API, tensor layouts and
return values are **unchanged** with respect to v2.

Requires PyTorch ≥ 2 .0 (validated on 2 .7 .0).
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple, Dict, Union

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

    def __init__(self, board_size: int = 19, device: Optional[torch.device] = None):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device

        # Re-use a single board checker instance – now *fully* batch-safe.
        self._checker = VectorizedBoardChecker(board_size, self.device)

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def compute_legal_moves_with_captures(
        self,
        stones: torch.Tensor,              # [B, 2, H, W]
        current_player: torch.Tensor,      # [B]
        ko_points: Optional[torch.Tensor] = None,  # [B, 2]
        return_capture_info: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Return legal-move mask (and capture meta-data if requested)."""
        B, _, H, W = stones.shape

        # Heavy lifting
        legal_mask, capture_info = self._checker.compute_batch_legal_and_captures(
            stones, current_player)

        # Ko rule (cheap)
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
    """Avoids all Python-level loops across the batch."""

    # --------------------------------------------------------------
    # Construction helpers
    # --------------------------------------------------------------
    def __init__(self, board_size: int, device: torch.device):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device
        self._init_neighbor_structure()

    def _init_neighbor_structure(self) -> None:
        """Pre-compute neighbour indices & validity mask."""
        Bsz   = self.board_size
        OFF   = torch.tensor([-Bsz, Bsz, -1, 1], dtype=IDX_DTYPE, device=self.device)
        flat  = torch.arange(self.N2, dtype=IDX_DTYPE, device=self.device)
        nbrs  = flat[:, None] + OFF                      # [N²,4]
        valid = (nbrs >= 0) & (nbrs < self.N2)
        col   = flat % Bsz
        valid[:, 2] &= col != 0                          # W edge
        valid[:, 3] &= col != Bsz - 1                    # E edge
        self.NEIGH_IDX   = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.NEIGH_VALID = valid

    # --------------------------------------------------------------
    # Top-level batched computation
    # --------------------------------------------------------------
    def compute_batch_legal_and_captures(
        self,
        stones: torch.Tensor,         # [B, 2, H, W]
        current_player: torch.Tensor  # [B]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, _, H, W = stones.shape
        N2 = self.N2
        stones_f = stones.view(B, 2, N2)
        occupied = stones_f.any(dim=1)        # [B,N²]
        empty    = ~occupied

        # Build union-find structures (fully batched)
        _, colour, roots, root_libs = self._batch_init_union_find(stones_f)

        curr_player = current_player.view(B, 1)
        opp_player  = 1 - curr_player

        # Neighbour lookups
        neigh_colors = self._get_neighbor_colors_batch(colour)   # [B,N²,4]
        neigh_roots  = self._get_neighbor_roots_batch(roots)     # [B,N²,4]
        valid_mask   = self.NEIGH_VALID.view(1, N2, 4)

        # Immediate liberties
        has_any_lib = ((neigh_colors == -1) & valid_mask).any(dim=2)

        # Capture detection
        opp_mask = (neigh_colors == opp_player.view(B, 1, 1)) & valid_mask
        neigh_roots_flat = neigh_roots.reshape(B, -1)
        neigh_libs_flat  = root_libs.gather(1, neigh_roots_flat.clamp(min=0))
        neigh_libs       = neigh_libs_flat.view(B, N2, 4)
        can_capture      = opp_mask & (neigh_libs == 1)
        can_capture_any  = can_capture.any(dim=2)

        # Friendly connection with liberties
        friendly      = (neigh_colors == curr_player.view(B, 1, 1)) & valid_mask
        friendly_libs = friendly & (neigh_libs > 1)
        friendly_any  = friendly_libs.any(dim=2)

        # Final legality mask
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)

        # Capture meta-data
        capture_groups = torch.full((B, N2, 4), -1, dtype=IDX_DTYPE, device=self.device)
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
    # Batch helpers (no Python loops)
    # --------------------------------------------------------------
    def _get_neighbor_colors_batch(self, colour: torch.Tensor) -> torch.Tensor:
        B, N2 = colour.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)

        idx_flat   = idx.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)
        neigh_flat = torch.full_like(idx_flat, -2, dtype=DTYPE)
        g = torch.gather(colour, 1, idx_flat.clamp(min=0))
        neigh_flat[valid_flat] = g[valid_flat]
        return neigh_flat.view(B, N2, 4)

    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        B, N2 = roots.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)

        idx_flat   = idx.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)
        neigh_flat = torch.full_like(idx_flat, -1, dtype=IDX_DTYPE)
        g = torch.gather(roots, 1, idx_flat.clamp(min=0))
        neigh_flat[valid_flat] = g[valid_flat]
        return neigh_flat.view(B, self.N2, 4)

    # --------------------------------------------------------------
    # Fully batched union-find initialisation (no Python loop over B)
    # --------------------------------------------------------------
    def _batch_init_union_find(
        self, stones_f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, N2 = stones_f.shape
        dev = self.device

        # Colours and initial parent pointers
        colour = torch.full((B, N2), -1, dtype=DTYPE, device=dev)
        colour[stones_f[:, 0].bool()] = 0
        colour[stones_f[:, 1].bool()] = 1

        parent = torch.arange(N2, dtype=IDX_DTYPE, device=dev).repeat(B, 1)

        # Merge same-colour neighbours (batched)
        neigh_cols = self._get_neighbor_colors_batch(colour)              # [B,N²,4]
        same = (neigh_cols == colour.unsqueeze(2)) \
               & (colour.unsqueeze(2) != -1) \
               & self.NEIGH_VALID.view(1, N2, 4)

        if same.any():
            b, p, d = same.nonzero(as_tuple=True)
            nbr     = self.NEIGH_IDX[p, d]
            gpos    = b * N2 + p
            gnbr    = b * N2 + nbr
            parent_flat = parent.view(-1)
            new_par = torch.minimum(parent_flat[gpos], parent_flat[gnbr])
            parent_flat[gpos] = new_par
            parent_flat[gnbr] = new_par
            # Few halving passes for compression
            for _ in range(81):
                parent_flat[:] = parent_flat[parent_flat]
            parent = parent_flat.view(B, N2)

        roots = parent.clone()

        # Count unique liberties per root (batched)
        neigh_cols = self._get_neighbor_colors_batch(colour)
        is_lib = (neigh_cols == -1) & self.NEIGH_VALID.view(1, N2, 4)
        stone_mask = colour != -1

        libs_per_root = torch.zeros(B * N2, dtype=IDX_DTYPE, device=dev)
        if stone_mask.any():
            batch_map = torch.arange(B, device=dev, dtype=IDX_DTYPE).view(B, 1, 1)
            roots_exp = roots.unsqueeze(2)
            lib_idx   = self.NEIGH_IDX.view(1, N2, 4)
            mask      = is_lib & stone_mask.unsqueeze(2)

            fb = batch_map.expand_as(mask)[mask]
            fr = roots_exp.expand_as(mask)[mask]
            fl = lib_idx.expand_as(mask)[mask]

            key_root = fb * N2 + fr
            key_lib  = fb * N2 + fl
            pair = torch.stack((key_root, key_lib), 1)
            pair = pair[pair[:, 1].argsort()]
            uniq = torch.unique_consecutive(pair, dim=0)
            libs_per_root.scatter_add_(0, uniq[:, 0], torch.ones_like(uniq[:, 0]))

        root_libs = libs_per_root.view(B, N2)
        return parent, colour, roots, root_libs
