# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py – board-plane edition (perf/readability pass)
===================================================================

Drop-in replacement for your existing module. Same public API, clearer names,
explicit shapes, and a batched, row-safe pointer-jumping step.

Perf tweaks in this version:
- Row-local ids (parent/roots/lib-counts) use int32 to cut bandwidth.
- Gather indices remain int64 (PyTorch requirement).
- Allocation-free union-find loop (masked_fill with scalar sentinel).
- Tighter UF iteration cap: O(log N) with periodic convergence checks.

Adds CSR capture lists in a flat row-space:
- cap_indptr: (B*N2 + 1) int32
- cap_indices: (L,)      int32
while still returning only light metadata (legacy dense mask commented out).

Board
-----
- board: (B, H, W) int8 with values: -1 empty, 0 black, 1 white
- Internally we work on a flattened grid: N2 = H * W
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch

# -----------------------------------------------------------------------------
# Dtypes & sentinels
# -----------------------------------------------------------------------------
DTYPE_COLOR: torch.dtype = torch.int8
IDX_DTYPE_ROW: torch.dtype = torch.int32   # row-local ids [0..N2)
IDX_DTYPE_GTH: torch.dtype = torch.int64   # indices passed to gather / advanced indexing
KEY_DTYPE:     torch.dtype = torch.int64   # flattened keys (b*N2 + i / root)

SENTINEL_NEIGH_COLOR = -2  # off-board neighbour color fill (int8 ok)
SENTINEL_NEIGH_ROOT  = -1  # off-board neighbour root fill (fits int32/64)


# =============================================================================
# Public API
# =============================================================================
class GoLegalMoveChecker:
    def __init__(self, board_size=19, device=None, enable_csr: bool = True):
        self._checker = VectorizedBoardChecker(board_size, device, enable_csr=enable_csr)

    @torch.inference_mode()
    def compute_batch_legal_and_info(self, board, current_player, return_info=True):
        B, H, W = board.shape
        assert H == W == self._checker.board_size, "board size mismatch"
        legal, info = self._checker.compute_batch_legal_and_info(board, current_player)
        return (legal, info) if return_info else legal


# =============================================================================
# Batched implementation
# =============================================================================
class VectorizedBoardChecker:
    """
    Fully batched legal-move logic with capture detection.
    Works on (B,H,W) int8 boards, flattens to (B,N2) internally.
    """

    # Per-board (flatten) structures
    index_flatten: torch.Tensor          # (N2,) int32 row-local ids
    neigh_index_flatten: torch.Tensor    # (N2,4) int64 — gather index table
    neigh_valid_flatten: torch.Tensor    # (N2,4) bool

    # Per-call (runtime) data
    board_flatten: torch.Tensor          # (B,N2), set each call

    def __init__(self, board_size: int, device: Optional[torch.device], enable_csr: bool = True):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device
        self.enable_csr = enable_csr
        self._init_board_flatten_structure()

    # ------------------------------------------------------------------
    # Precomputed 4-neighbour tables in flat space
    # ------------------------------------------------------------------
    def _init_board_flatten_structure(self) -> None:
        N, N2, dev = self.board_size, self.N2, self.device

        # (N2,) flat indices of a single board (row-local id space)
        self.index_flatten = torch.arange(N2, dtype=IDX_DTYPE_ROW, device=dev)

        # (N2,4) neighbours via offsets: N,S,W,E — keep LONG for gather
        OFF  = torch.tensor([-N, N, -1, 1], dtype=IDX_DTYPE_GTH, device=dev)      # (4,)
        nbrs = self.index_flatten.to(IDX_DTYPE_GTH)[:, None] + OFF                # (N2,4) long

        # Edge handling
        valid = (nbrs >= 0) & (nbrs < N2)                                         # (N2,4) bool
        col   = (self.index_flatten % N).to(IDX_DTYPE_GTH)
        valid[:, 2] &= col != 0           # W invalid at left edge
        valid[:, 3] &= col != N - 1       # E invalid at right edge

        self.neigh_index_flatten = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.neigh_valid_flatten = valid

    # ------------------------------------------------------------------
    # Top-level: legal mask + capture metadata (dense + optional CSR)
    # ------------------------------------------------------------------
    def compute_batch_legal_and_info(
        self,
        board: torch.Tensor,          # (B,H,W) int8 in {-1,0,1}
        current_player: torch.Tensor  # (B,)    uint8 in {0,1}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dev = self.device
        B, H, W = board.shape
        N2      = self.N2

        # Per-call runtime flatten (depends on B)
        self.board_flatten = board.reshape(B, N2)         # (B,N2)
        empty              = (self.board_flatten == -1)   # (B,N2) bool

        # Groups (roots) + liberties
        roots, root_libs = self._batch_init_union_find()  # roots: int32, root_libs: int32 (B,N2)

        # Batched neighbour tables
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)

        # Neighbour colors/roots (colors are read from self.board_flatten)
        neigh_colors = self._get_neighbor_colors_batch()      # (B,N2,4) int8/sentinels
        neigh_roots  = self._get_neighbor_roots_batch(roots)  # (B,N2,4) int32/sentinels

        curr = current_player.view(B, 1, 1)  # (B,1,1)
        opp  = 1 - curr

        # A) immediate liberties
        has_any_lib = ((neigh_colors == -1) & neigh_valid_flatten_b).any(dim=2)   # (B,N2)

        # B) captures: adjacent opponent group with exactly 1 liberty
        neigh_roots_f = neigh_roots.reshape(B, -1)                                # (B,N2*4) int32
        # Gather needs long indices
        neigh_roots_f_long = neigh_roots_f.clamp(min=0).to(torch.long)            # (B,N2*4) long
        neigh_libs_f  = root_libs.to(torch.long).gather(1, neigh_roots_f_long)    # (B,N2*4) long
        neigh_libs    = neigh_libs_f.view(B, N2, 4).to(IDX_DTYPE_ROW)             # (B,N2,4) int32

        opp_mask        = (neigh_colors == opp) & neigh_valid_flatten_b           # (B,N2,4) bool
        can_capture     = opp_mask & (neigh_libs == 1)                             # (B,N2,4)
        can_capture_any = can_capture.any(dim=2)                                   # (B,N2)

        # C) friendly safe attachment
        friendly     = (neigh_colors == curr) & neigh_valid_flatten_b              # (B,N2,4)
        friendly_any = (friendly & (neigh_libs > 1)).any(dim=2)                    # (B,N2)

        # D) final legality
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)                                      # (B,H,W)

        # Base info (always include UF outputs)
        info: Dict[str, torch.Tensor] = {
            "roots": roots,                          # (B,N2) int32
            "root_libs": root_libs,                  # (B,N2) int32
            "can_capture_any": can_capture_any,      # (B,N2) bool
        }

        # ----------------------------------------------------------------------
        # CSR capture lists (flat row-space): cap_indptr (B*N2+1), cap_indices (L,)
        # Row id is r = b*N2 + i
        # ----------------------------------------------------------------------
        opp_colour = 1 - current_player.view(B, 1)
        is_opp    = (self.board_flatten == opp_colour).contiguous()               # (B,N2) bool

        b_flat  = torch.arange(B,  device=dev, dtype=KEY_DTYPE).repeat_interleave(N2)  # (B*N2,) int64
        j_flat  = torch.arange(N2, device=dev, dtype=KEY_DTYPE).repeat(B)              # (B*N2,) int64
        r_flat  = roots.reshape(-1).to(KEY_DTYPE)                                      # (B*N2,) int64
        keep    = is_opp.reshape(-1)                                                   # (B*N2,) bool

        key_root = b_flat[keep] * N2 + r_flat[keep]    # (Kopp,) int64
        j_list   = j_flat[keep]                        # (Kopp,) int64

        if key_root.numel() > 0:
            order        = key_root.argsort()
            key_sorted   = key_root[order]                 # (Kopp,)
            j_sorted     = j_list[order]                   # (Kopp,)
            uniq, counts = torch.unique_consecutive(key_sorted, return_counts=True)
            ends         = torch.cumsum(counts, 0)         # (G,)
            starts       = torch.empty_like(ends)
            starts[0]    = 0
            starts[1:]   = ends[:-1]

            # Map (b, root) -> [start,end)
            root_b = (uniq // N2).to(IDX_DTYPE_ROW)     # (G,) int32
            root_r = (uniq %  N2).to(IDX_DTYPE_ROW)     # (G,) int32
            root_start = torch.full((B, N2), -1, dtype=IDX_DTYPE_ROW, device=dev)
            root_end   = torch.full((B, N2), -1, dtype=IDX_DTYPE_ROW, device=dev)
            root_start[root_b, root_r] = starts.to(IDX_DTYPE_ROW)
            root_end[root_b, root_r]   = ends.to(IDX_DTYPE_ROW)
        else:
            j_sorted   = torch.empty(0, dtype=KEY_DTYPE, device=dev)
            root_start = torch.full((B, N2), -1, dtype=IDX_DTYPE_ROW, device=dev)
            root_end   = torch.full((B, N2), -1, dtype=IDX_DTYPE_ROW, device=dev)

        # For each candidate move, which neighbor roots are captured (≤4), dedup per move
        cand_roots = torch.full((B, N2, 4), SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE_ROW, device=dev)
        cand_roots[can_capture] = neigh_roots[can_capture]  # (B, N2, 4) int32

        # Sort per-move so duplicates sit next to each other
        roots_sorted, _ = cand_roots.sort(dim=2)           # (B, N2, 4) int32

        # Valid entries (not -1)
        valid = roots_sorted >= 0                           # (B, N2, 4)

        # Dedup within each move: keep first of any equal run along last axis
        same_as_prev = torch.zeros_like(valid)
        same_as_prev[..., 1:] = (roots_sorted[..., 1:] == roots_sorted[..., :-1]) & valid[..., 1:] & valid[..., :-1]
        unique_mask = valid & ~same_as_prev                 # unique roots per move

        # Gather start/end of each root’s member slice from the global maps.
        idx_flat = roots_sorted.clamp_min(0).reshape(B, -1).to(torch.long)  # long for gather
        lo_flat  = torch.gather(root_start.to(torch.long), 1, idx_flat)     # (B, N2*4) long
        hi_flat  = torch.gather(root_end.to(torch.long),   1, idx_flat)     # (B, N2*4) long
        lo = lo_flat.view(B, N2, 4).to(IDX_DTYPE_ROW)                        # (B, N2, 4) int32
        hi = hi_flat.view(B, N2, 4).to(IDX_DTYPE_ROW)                        # (B, N2, 4) int32

        # Segment lengths per captured root (0 where invalid/duplicate)
        zero_i32 = torch.zeros((), dtype=IDX_DTYPE_ROW, device=dev)
        seg_len = torch.where(unique_mask & (lo >= 0) & (hi > lo), hi - lo, zero_i32).to(IDX_DTYPE_ROW)  # (B,N2,4)

        # Row-local counts and GLOBAL flat indptr (length R+1)
        cap_counts      = seg_len.sum(dim=2).to(torch.int32)         # (B, N2)
        cap_counts_flat = cap_counts.reshape(-1)                     # (R,)
        R = B * N2

        cap_indptr_flat = torch.empty(R + 1, dtype=torch.int32, device=dev)
        cap_indptr_flat[0] = 0
        torch.cumsum(cap_counts_flat.to(torch.int32), 0, out=cap_indptr_flat[1:])  # row-major order

        total_L = int(cap_indptr_flat[-1].item())
        cap_indices = torch.empty(total_L, dtype=torch.int32, device=dev)

        # Fill cap_indices by concatenating member slices per (b,i) root — vectorized
        keep_trip = seg_len > 0                       # (B, N2, 4) bool

        # Build per-move identifiers
        b_ar = torch.arange(B,  device=dev, dtype=KEY_DTYPE).view(B, 1, 1).expand_as(seg_len)   # int64
        i_ar = torch.arange(N2, device=dev, dtype=KEY_DTYPE).view(1, N2, 1).expand_as(seg_len)  # int64

        b_tri   = b_ar[keep_trip]                     # (T,) int64
        i_tri   = i_ar[keep_trip]                     # (T,) int64
        lo_tri  = lo[keep_trip].to(KEY_DTYPE)         # (T,) int64
        len_tri = seg_len[keep_trip].to(KEY_DTYPE)    # (T,) int64

        # Row id in flat space
        row_flat = (b_tri * N2 + i_tri)               # (T,) int64
        base_row = cap_indptr_flat.to(KEY_DTYPE)[row_flat]  # (T,) int64

        # Sort triples by row to make per-row concatenation contiguous
        sort_key, ord = torch.sort(row_flat)
        row_flat = row_flat[ord]; base_row = base_row[ord]
        lo_tri   = lo_tri[ord];   len_tri  = len_tri[ord]

        # Running offsets within each row (0, n1, n1+n2, …)
        same_prev = torch.zeros_like(sort_key, dtype=torch.bool)
        same_prev[1:] = (sort_key[1:] == sort_key[:-1])
        csum  = torch.cumsum(len_tri, dim=0)                      # (T,) int64
        row_offset = torch.zeros_like(csum)
        row_offset[1:] = torch.where(same_prev[1:], csum[:-1], torch.zeros_like(csum[:-1]))

        # Element-level expansion (safe even when K == 0)
        K = int(len_tri.sum().item())
        if K > 0:
            elem_pos   = torch.arange(K, device=dev, dtype=KEY_DTYPE)                           # (K,)
            off_in_run = elem_pos - torch.repeat_interleave(row_offset, len_tri)                # (K,)

            src_idx = torch.repeat_interleave(lo_tri, len_tri) + off_in_run                    # (K,)
            dst_idx = (torch.repeat_interleave((base_row + row_offset), len_tri) + off_in_run) # (K,)

            cap_indices[dst_idx.to(torch.long)] = j_sorted[src_idx.to(torch.long)].to(torch.int32)

        # Publish CSR (flat row-space)
        info["cap_indptr"]  = cap_indptr_flat    # (B*N2 + 1,) int32
        info["cap_indices"] = cap_indices        # (L,) int32

        return legal_mask, info

    # ------------------------------------------------------------------
    # Union-find + liberties (flat graph; row-safe compression)
    # ------------------------------------------------------------------
    def _batch_init_union_find(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        roots     : (B,N2) int32  union-find representative per point
        root_libs : (B,N2) int32  liberty count per root id (index by root id)
        """
        dev = self.device
        board_flatten = self.board_flatten
        B, N2 = board_flatten.shape

        # Neighbour colors (computed from board_flatten directly)
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)
        neigh_cols            = self._get_neighbor_colors_batch()  # (B,N2,4)

        # Same-color adjacency (ignore empties; respect edges)
        same = (neigh_cols == board_flatten.unsqueeze(2)) \
             & (board_flatten.unsqueeze(2) != -1) \
             & neigh_valid_flatten_b                                              # (B,N2,4)

        # Hook & compress (row-local ids int32; indices for gather are int64)
        parent = torch.arange(N2, dtype=IDX_DTYPE_ROW, device=dev).expand(B, N2)  # (B,N2) int32
        parent = self._hook_and_compress(parent, same)                            # (B,N2) int32
        roots  = parent                                                           # (B,N2) int32

        # Count unique liberties per root
        is_lib     = (neigh_cols == -1) & neigh_valid_flatten_b                   # (B,N2,4)
        stone_mask = (board_flatten != -1)                                        # (B,N2)

        libs_per_root = torch.zeros(B * N2, dtype=IDX_DTYPE_ROW, device=dev)      # (B*N2,) int32

        mask = is_lib & stone_mask.unsqueeze(2)                                   # (B,N2,4)

        # edges → pairs (root_key, liberty_key) for scatter dedup
        # mask: (B,N2,4)
        b_idx, i_idx, d_idx = torch.where(mask)                     # each (K,) long
        fr = roots[b_idx, i_idx].to(KEY_DTYPE)                      # (K,) int64
        fb = b_idx.to(KEY_DTYPE)                                    # (K,) int64
        fl = self.neigh_index_flatten[i_idx.to(torch.long), d_idx.to(torch.long)].to(KEY_DTYPE)  # (K,) int64

        key_root = fb * N2 + fr
        key_lib  = fb * N2 + fl
        pairs    = torch.stack((key_root, key_lib), dim=1)                         # (K,2) int64

        sort_key     = pairs[:, 0] * (N2 * B) + pairs[:, 1]
        sorted_idx   = sort_key.argsort()
        pairs_sorted = pairs[sorted_idx]
        uniq         = torch.unique_consecutive(pairs_sorted, dim=0)

        # Each unique (root,liberty) contributes 1 liberty
        libs_per_root = libs_per_root.to(torch.long)  # scatter_add requires dtypes to match indices
        libs_per_root.scatter_add_(0, uniq[:, 0], torch.ones_like(uniq[:, 0], dtype=torch.long))
        root_libs = libs_per_root.to(IDX_DTYPE_ROW).reshape(B, N2)                # (B,N2) int32

        return roots, root_libs

    # ------------------------------------------------------------------
    # Batched pointer-jumping with periodic convergence check
    # ------------------------------------------------------------------
    def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        """
        parent: (B,N2) int32 (row-local ids)
        same  : (B,N2,4) bool adjacency mask for same-color
        """
        B, N2 = parent.shape
        dev = parent.device

        # Gather indices must be long; keep a single expanded tensor
        nbr_idx_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp_min(0)  # (B,N2,4) long

        # Tighter iteration cap; pointer-jumping is ~log N
        max_rounds = (N2.bit_length()) + 6

        for i in range(max_rounds):
            parent_prev = parent

            # Gather neighbors' parents
            nbr_parent = torch.gather(parent, 1, nbr_idx_b.reshape(B, -1)) \
                           .view(B, N2, 4)                                     # (B,N2,4) int32
            # Avoid per-iter allocations: fill scalar where !same
            nbr_parent = nbr_parent.masked_fill(~same, N2)                      # N2 is a safe "big" sentinel

            min_nbr = nbr_parent.min(dim=2).values                               # (B,N2) int32
            hooked  = torch.minimum(parent, min_nbr)                              # (B,N2) int32

            # Two jumps per loop
            comp = torch.gather(hooked, 1, hooked.to(torch.long))                # parent[parent]
            comp = torch.gather(comp,   1, comp.to(torch.long))                  # parent[parent[parent]]

            # Lazy convergence check (every 8 iters)
            if (i & 3) == 3 and torch.equal(comp, parent_prev):
                return comp
            parent = comp
        return parent

    # ------------------------------------------------------------------
    # Neighbour helpers (batched, flat graph, 4 dirs)
    # ------------------------------------------------------------------
    def _get_neighbor_colors_batch(self) -> torch.Tensor:
        """Return neighbor colors(stones) pulled from self.board_flatten."""
        B, N2 = self.board_flatten.shape
        neigh_index_f_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)  # (B,N2,4) long
        neigh_valid_f_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)               # (B,N2,4) bool

        board3   = self.board_flatten.unsqueeze(2).expand(-1, -1, 4)                               # (B,N2,4) int8
        gathered = torch.gather(board3, dim=1, index=neigh_index_f_b)                              # (B,N2,4) int8

        out = torch.where(
            neigh_valid_f_b,
            gathered.to(DTYPE_COLOR),
            torch.full_like(gathered, SENTINEL_NEIGH_COLOR, dtype=DTYPE_COLOR)
        )
        return out  # (B,N2,4) int8

    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        """
        roots: (B,N2) int32 (row-local ids)
        returns: (B,N2,4) int32 neighbor roots with SENTINEL_NEIGH_ROOT for off-board
        """
        B, N2 = roots.shape
        neigh_index_f_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)  # (B,N2,4) long
        neigh_valid_f_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)               # (B,N2,4) bool

        roots3   = roots.unsqueeze(2).expand(-1, -1, 4)                                            # (B,N2,4) int32
        gathered = torch.gather(roots3, dim=1, index=neigh_index_f_b)                              # (B,N2,4) int32

        out = torch.where(
            neigh_valid_f_b,
            gathered,
            torch.full_like(gathered, SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE_ROW)
        )
        return out  # (B,N2,4) int32
