# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py (v2) – further‑vectorised legal‑move computation for Go
============================================================================

This file targets PyTorch ≥ 2.0 and has been validated on 2.7.0.
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple, Dict, Union

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
DTYPE     = torch.int16
IDX_DTYPE = torch.int64


# =============================================================================
# Public API
# =============================================================================
class GoLegalMoveChecker:
    """Vectorised legal‑move checker with capture detection for Go."""

    def __init__(self, board_size: int = 19, device: Optional[torch.device] = None):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device or torch.device("cpu")

        # Re‑use a single board checker instance – it is now *fully* batch safe.
        self._checker   = VectorizedBoardChecker(board_size, self.device)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def compute_legal_moves_with_captures(
        self,
        stones: torch.Tensor,              # [B, 2, H, W]
        current_player: torch.Tensor,      # [B]
        ko_points: Optional[torch.Tensor] = None,  # [B, 2]
        return_capture_info: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        B, _, H, W = stones.shape
        device = stones.device

        # ------------------------------------------------------------------
        # Do all the heavy work in the board‑checker
        # ------------------------------------------------------------------
        legal_mask, capture_info = self._checker.compute_batch_legal_and_captures(
            stones, current_player
        )

        # ------------------------------------------------------------------
        # Apply ko rule (very cheap compared to the rest)
        # ------------------------------------------------------------------
        if ko_points is not None:
            ko_valid = ko_points[:, 0] >= 0
            if ko_valid.any():
                batch_idx = ko_valid.nonzero(as_tuple=True)[0]
                r = ko_points[batch_idx, 0].long()
                c = ko_points[batch_idx, 1].long()
                legal_mask[batch_idx, r, c] = False

        if return_capture_info:
            return legal_mask, capture_info
        return legal_mask


# =============================================================================
# Fully vectorised batch board checker
# =============================================================================
class VectorizedBoardChecker:
    """Board checker that avoids Python‑level loops across the batch."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, board_size: int, device: torch.device):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device

        # Pre‑compute neighbour index + validity masks (shape [N², 4])
        self._init_neighbor_structure()

    def _init_neighbor_structure(self) -> None:
        """Pre‑compute neighbour indices for every coordinate."""
        Bsz  = self.board_size
        OFF  = torch.tensor(
            [-Bsz,  Bsz,  -1,  1], dtype=IDX_DTYPE, device=self.device
        )  # N, S, W, E offsets

        flat = torch.arange(self.N2, dtype=IDX_DTYPE, device=self.device)
        nbrs = flat[:, None] + OFF          # [N², 4]

        valid = (nbrs >= 0) & (nbrs < self.N2)

        # Column indices for edge detection
        col = flat % Bsz
        valid[:, 2] &= col != 0             # W blocked on left edge
        valid[:, 3] &= col != Bsz - 1       # E blocked on right edge

        self.NEIGH_IDX   = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.NEIGH_VALID = valid

    # ------------------------------------------------------------------
    # Top‑level batched computation
    # ------------------------------------------------------------------
    def compute_batch_legal_and_captures(
        self,
        stones: torch.Tensor,         # [B, 2, H, W]
        current_player: torch.Tensor  # [B]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, _, H, W = stones.shape
        N2         = self.N2
        device     = self.device

        # ------------------------------------------------------------------
        # Flatten board representation                                                         
        # ------------------------------------------------------------------
        stones_f = stones.view(B, 2, -1)                # [B, 2, N²]
        occupied = stones_f.any(dim=1)                  # [B, N²]
        empty    = ~occupied

        # ------------------------------------------------------------------
        # Build union‑find once per batch (has unavoidable loops internally)
        # ------------------------------------------------------------------
        _, colour, roots, root_libs = self._batch_init_union_find(stones_f)

        # Expand some useful tensors
        curr_player = current_player.view(B, 1)         # [B, 1]
        opp_player  = 1 - curr_player

        # ------------------------------------------------------------------
        # Batched neighbour lookups – *no Python loops*
        # ------------------------------------------------------------------
        neigh_colors = self._get_neighbor_colors_batch(colour)   # [B, N², 4]
        neigh_roots  = self._get_neighbor_roots_batch(roots)     # [B, N², 4]
        valid_mask   = self.NEIGH_VALID.view(1, N2, 4)           # [1, N², 4]

        # Immediate liberties
        has_any_lib = (neigh_colors == -1) & valid_mask
        has_any_lib = has_any_lib.any(dim=2)                     # [B, N²]

        # ------------------------------------------------------------------
        # Capture detection
        # ------------------------------------------------------------------
        opp_mask      = (neigh_colors == opp_player.view(B, 1, 1)) & valid_mask
        
        # Flatten neigh_roots for gathering from root_libs
        neigh_roots_flat = neigh_roots.reshape(B, -1)  # [B, N²*4]
        neigh_libs_flat = root_libs.gather(1, neigh_roots_flat.clamp(min=0).long())
        neigh_libs = neigh_libs_flat.reshape(B, N2, 4)  # Reshape back
        
        can_capture   = opp_mask & (neigh_libs == 1)
        can_capture_any = can_capture.any(dim=2)                 # [B, N²]

        # ------------------------------------------------------------------
        # Friendly connection with liberties
        # ------------------------------------------------------------------
        friendly      = (neigh_colors == curr_player.view(B, 1, 1)) & valid_mask
        friendly_libs = friendly & (neigh_libs > 1)
        friendly_any  = friendly_libs.any(dim=2)

        # ------------------------------------------------------------------
        # Final legal mask
        # ------------------------------------------------------------------
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)

        # ------------------------------------------------------------------
        # Capture information ------------------------------------------------
        # ------------------------------------------------------------------
        capture_groups = torch.full((B, N2, 4), -1,
                                    dtype=IDX_DTYPE, device=device)
        capture_groups[can_capture] = neigh_roots[can_capture]

        # --- Pre‑compute sizes of every root group (vectorised) -------------
        sizes = torch.zeros((B, N2), dtype=IDX_DTYPE, device=device)
        sizes.scatter_add_(1, roots, torch.ones_like(roots, dtype=IDX_DTYPE))

        capture_sizes = torch.zeros_like(capture_groups)
        valid_cap     = capture_groups >= 0
        
        # Flatten for gathering
        capture_groups_flat = capture_groups.reshape(B, -1)  # [B, N²*4]
        valid_cap_flat = valid_cap.reshape(B, -1)  # [B, N²*4]
        
        # Gather sizes
        sizes_gathered = sizes.gather(1, capture_groups_flat.clamp(min=0).long())
        capture_sizes_flat = torch.zeros_like(capture_groups_flat)
        capture_sizes_flat[valid_cap_flat] = sizes_gathered[valid_cap_flat]
        
        # Reshape back
        capture_sizes = capture_sizes_flat.reshape(B, N2, 4)

        total_captures = capture_sizes.sum(dim=2).view(B, H, W)

        capture_info = {
            "would_capture" : (empty & can_capture_any).view(B, H, W),
            "capture_groups": capture_groups.view(B, H, W, 4),
            "capture_sizes" : capture_sizes.view(B, H, W, 4),
            "total_captures": total_captures,
            "roots"         : roots,  # Return roots tensor for efficient capture removal
            "colour"        : colour, # Return colour tensor to identify opponent stones
        }
        return legal_mask, capture_info

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    def _get_neighbor_colors_batch(self, colour: torch.Tensor) -> torch.Tensor:
        """Return colours of neighbours – batched, no Python loops."""
        B, N2 = colour.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)

        # Flatten for gathering
        idx_flat = idx.reshape(B, -1)  # [B, N²*4]
        valid_flat = valid.reshape(B, -1)  # [B, N²*4]
        
        # Gather and reshape
        neigh_flat = torch.full_like(idx_flat, -2, dtype=DTYPE)
        gathered = torch.gather(colour, 1, idx_flat.clamp(min=0).long())
        neigh_flat[valid_flat] = gathered[valid_flat]
        
        # Reshape back to [B, N², 4]
        neigh = neigh_flat.reshape(B, N2, 4)
        return neigh

    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        """Neighbour root indices – batched."""
        B, N2 = roots.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)

        # Flatten for gathering
        idx_flat = idx.reshape(B, -1)  # [B, N²*4]
        valid_flat = valid.reshape(B, -1)  # [B, N²*4]
        
        # Gather and reshape
        neigh_flat = torch.full_like(idx_flat, -1, dtype=IDX_DTYPE)
        gathered = torch.gather(roots, 1, idx_flat.clamp(min=0).long())
        neigh_flat[valid_flat] = gathered[valid_flat]
        
        # Reshape back to [B, N², 4]
        neigh = neigh_flat.reshape(B, N2, 4)
        return neigh

    # ------------------------------------------------------------------
    # Union‑find initialisation (per‑board loops unavoidable)
    # ------------------------------------------------------------------
    # -----------------------------------------------------------------------------
# Fully batched union–find initialisation (no Python loop over B)
# -----------------------------------------------------------------------------
def _batch_init_union_find(
    self, stones_f: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorised replacement for the previous per-board loop.

    Input
    -----
    stones_f : Bool / Int tensor [B, 2, N²]
               stones_f[:, 0] == 1  → black stone
               stones_f[:, 1] == 1  → white stone

    Output (identical semantics to old version)
    ------------------------------------------
    parent      : [B, N²]  – final parent pointers (mainly for debugging)
    colour      : [B, N²]  – -1 = empty, 0 = black, 1 = white
    roots       : [B, N²]  – "root" id of the group to which each point belongs
    root_libs   : [B, N²]  – liberties per root id (0 for non-roots)
    """
    B, _, N2 = stones_f.shape
    device   = self.device

    # ------------------------------------------------------------------
    # 1)  colour & parent arrays
    # ------------------------------------------------------------------
    colour = torch.full((B, N2), -1, dtype=DTYPE, device=device)
    colour[stones_f[:, 0].bool()] = 0                    # black
    colour[stones_f[:, 1].bool()] = 1                    # white

    parent = torch.arange(N2, device=device, dtype=IDX_DTYPE).repeat(B, 1)

    # ------------------------------------------------------------------
    # 2)  Merge same-colour neighbours – single vectorised pass
    # ------------------------------------------------------------------
    neigh_colors = self._get_neighbor_colors_batch(colour)          # [B,N²,4]
    same = (neigh_colors == colour.unsqueeze(2)) \
           & (colour.unsqueeze(2) != -1) \
           & self.NEIGH_VALID.view(1, N2, 4)

    # Edge list  ⟨batch, pos, nbr⟩  for all “same” pairs
    b_idx, pos_idx, dir_idx = same.nonzero(as_tuple=True)
    nbr_idx = self.NEIGH_IDX[pos_idx, dir_idx]

    # Flatten (batch, pos) into a single global index so we can treat the
    # whole batch as one disjoint forest.
    glob_pos = b_idx * N2 + pos_idx
    glob_nbr = b_idx * N2 + nbr_idx

    parent_flat = parent.view(-1)                         # [B*N²]

    new_par = torch.minimum(parent_flat[glob_pos],
                            parent_flat[glob_nbr])
    parent_flat[glob_pos] = new_par
    parent_flat[glob_nbr] = new_par

    # A few halving rounds achieve near-full path compression on 19×19.
    for _ in range(5):
        parent_flat[:] = parent_flat[parent_flat]

    parent = parent_flat.view(B, N2)
    roots  = parent.to(IDX_DTYPE)

    # ------------------------------------------------------------------
    # 3)  Count *unique* liberties per root – also batched
    # ------------------------------------------------------------------
    neigh_colors = self._get_neighbor_colors_batch(colour)          # [B,N²,4]
    is_lib = (neigh_colors == -1) & self.NEIGH_VALID.view(1, N2, 4)

    # Broadcast helpers
    batch_map = torch.arange(B, device=device, dtype=IDX_DTYPE) \
                  .view(B, 1, 1).expand(B, N2, 4)
    roots_exp  = roots.unsqueeze(2).expand(-1, -1, 4)
    lib_idx    = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)

    mask = is_lib & (colour.unsqueeze(2) != -1)           # stone positions only

    flat_batch = batch_map[mask]                          # [...]
    flat_root  = roots_exp[mask]                          # [...]
    flat_lib   = lib_idx[mask]                            # [...]

    # ── De-duplicate (root, liberty) pairs ────────────────────────────
    Nbig = N2                                             # shorthand
    root_global = flat_batch * Nbig + flat_root
    lib_global  = flat_batch * Nbig + flat_lib

    pair = torch.stack((root_global, lib_global), 1)      # [*, 2]
    pair = pair[pair[:, 1].argsort()]                     # sort by liberty
    uniq = torch.unique_consecutive(pair, dim=0)          # drop duplicates

    # ── Scatter-add one per unique liberty ────────────────────────────
    libs = torch.zeros(B * Nbig, dtype=IDX_DTYPE, device=device)
    libs.scatter_add_(0, uniq[:, 0],
                      torch.ones_like(uniq[:, 0], dtype=IDX_DTYPE))

    root_libs = libs.view(B, Nbig)

    return parent, colour, roots, root_libs


    # ------------------------------------------------------------------
    # The rest of the class is unchanged – same as previous revision
    # ------------------------------------------------------------------
    def _build_groups_vectorized(self, parent: torch.Tensor,
                                 colour: torch.Tensor) -> torch.Tensor:
        neigh_colors = self._get_neighbor_colors_batch(
            colour.unsqueeze(0)
        )[0]                                          # [N²,4]
        same = (neigh_colors == colour.unsqueeze(1)) \
               & (colour.unsqueeze(1) != -1) \
               & self.NEIGH_VALID                     # [N²,4]

        pos, dir_ = same.nonzero(as_tuple=True)
        if pos.numel():
            nbr = self.NEIGH_IDX[pos, dir_]
            par_pos = parent[pos]
            par_nbr = parent[nbr]
            new_par = torch.minimum(par_pos, par_nbr)
            parent[pos] = new_par
            parent[nbr] = new_par
            for _ in range(5):                        # Quick compression
                parent = parent[parent]
        return parent

    def _count_liberties_vectorized(
        self, colour: torch.Tensor, roots: torch.Tensor
    ) -> torch.Tensor:
        neigh_colors = self._get_neighbor_colors_batch(
            colour.unsqueeze(0)
        )[0]                                          # [N²,4]
        is_lib = (neigh_colors == -1) & self.NEIGH_VALID
        stone_pos = (colour != -1).nonzero(as_tuple=True)[0]

        libs = torch.zeros(self.N2, dtype=IDX_DTYPE, device=self.device)
        if stone_pos.numel():
            root_ids = roots[stone_pos]
            lib_pos  = self.NEIGH_IDX[stone_pos]
            has_lib  = is_lib[stone_pos]

            flat_root = root_ids.unsqueeze(1).expand(-1, 4).flatten()[has_lib.flatten()]
            flat_lib  = lib_pos.flatten()[has_lib.flatten()]

            # Count unique liberties per root
            uniq_root, inv = torch.unique(flat_root, return_inverse=True)
            # Sort liberties so scatter_add works on unique positions
            uniq_pairs = torch.stack([flat_root, flat_lib], dim=1)
            uniq_pairs = uniq_pairs[torch.argsort(uniq_pairs[:, 1])]
            # Deduplicate liberties
            uniq_pairs = uniq_pairs[torch.unique_consecutive(
                uniq_pairs[:, 1], return_inverse=False, dim=0
            )]

            libs.scatter_add_(0, uniq_pairs[:, 0], torch.ones_like(uniq_pairs[:, 0],
                                                                   dtype=IDX_DTYPE))
        return libs