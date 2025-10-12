# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py – board-plane edition (readability pass)
=============================================================

Drop-in replacement for your existing module. Same public API, clearer names,
explicit shapes, and a batched, row-safe pointer-jumping step.

Adds CSR capture lists in a flat row-space:
- cap_indptr: (B*N2 + 1) int32
- cap_indices: (L,)      int32
while still returning the legacy dense capture_stone_mask for migration.

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
IDX_DTYPE:   torch.dtype = torch.int64

SENTINEL_NEIGH_COLOR = -2  # off-board neighbour color fill
SENTINEL_NEIGH_ROOT  = -1  # off-board neighbour root fill


# =============================================================================
# Public API
# =============================================================================
class GoLegalMoveChecker:
    def __init__(self, board_size=19, device=None, enable_csr: bool = True):
        self._checker = VectorizedBoardChecker(board_size, device, enable_csr=enable_csr)

    @torch.inference_mode()
    def compute_batch_legal_and_info(self, board, current_player, return_info=True):
        B,H,W = board.shape
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
    index_flatten: torch.Tensor          # (N2,)
    neigh_index_flatten: torch.Tensor    # (N2,4) int64
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

        # (N2,) flat indices of a single board
        self.index_flatten = torch.arange(N2, dtype=IDX_DTYPE, device=dev)

        # (N2,4) neighbours via offsets: N,S,W,E
        OFF  = torch.tensor([-N, N, -1, 1], dtype=IDX_DTYPE, device=dev)    # (4,)
        nbrs = self.index_flatten[:, None] + OFF                             # (N2,4)

        # Edge handling
        valid = (nbrs >= 0) & (nbrs < N2)                                    # (N2,4)
        col   = self.index_flatten % N
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
        roots, root_libs = self._batch_init_union_find()  # (B,N2) each

        # Batched neighbour tables
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)

        # Neighbour colors/roots (colors are read from self.board_flatten)
        neigh_colors = self._get_neighbor_colors_batch()      # (B,N2,4)
        neigh_roots  = self._get_neighbor_roots_batch(roots)  # (B,N2,4)

        curr = current_player.view(B, 1, 1)  # (B,1,1)
        opp  = 1 - curr

        # A) immediate liberties
        has_any_lib = ((neigh_colors == -1) & neigh_valid_flatten_b).any(dim=2)   # (B,N2)

        # B) captures: adjacent opponent group with exactly 1 liberty
        neigh_roots_f = neigh_roots.reshape(B, -1)                                 # (B,N2*4)
        neigh_libs_f  = root_libs.gather(1, neigh_roots_f.clamp(min=0))            # (B,N2*4)
        neigh_libs    = neigh_libs_f.view(B, N2, 4)                                 # (B,N2,4)

        opp_mask        = (neigh_colors == opp) & neigh_valid_flatten_b             # (B,N2,4) bool
        can_capture     = opp_mask & (neigh_libs == 1)                              # (B,N2,4)
        can_capture_any = can_capture.any(dim=2)                                    # (B,N2)

        # C) friendly safe attachment
        friendly     = (neigh_colors == curr) & neigh_valid_flatten_b               # (B,N2,4)
        friendly_any = (friendly & (neigh_libs > 1)).any(dim=2)                     # (B,N2)

        # D) final legality
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)                                       # (B,H,W)

        # ----------------------------------------------------------------------
        # Dense capture mask (legacy) – keep during migration for placement
        # ----------------------------------------------------------------------
        capture_groups_dense = torch.full(
            (B, N2, 4), SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE, device=dev
        )  # (B,N2,4)
        capture_groups_dense = torch.where(can_capture, neigh_roots, capture_groups_dense)

        capture_groups_exp = capture_groups_dense.view(B, N2, 4, 1)                 # (B,N2,4,1)
        roots_exp          = roots.view(B, 1, 1, N2)                                # (B,1,1,N2)
        group_matches      = (capture_groups_exp == roots_exp) & (capture_groups_exp >= 0)
        capture_stone_mask = group_matches.any(dim=2)                                # (B,N2,N2)

        # Keep opponent stones only
        opp_colour         = 1 - current_player.view(B, 1)
        is_opponent_stone  = (self.board_flatten == opp_colour)                      # (B,N2)
        capture_stone_mask = capture_stone_mask & is_opponent_stone.view(B, 1, N2)

        # Belt-and-braces: cannot "capture" the just-placed stone itself here
        diag = torch.arange(N2, device=dev)
        capture_stone_mask[:, diag, diag] = False

        # Base info (always include legacy)
        info: Dict[str, torch.Tensor] = {
            "roots": roots,                          # (B,N2) int64
            "root_libs": root_libs,                  # (B,N2) int64
            "can_capture_any": can_capture_any,      # (B,N2) bool
            "capture_stone_mask": capture_stone_mask # (B,N2,N2) bool
        }

        if not self.enable_csr:
            return legal_mask, info

        # ----------------------------------------------------------------------
        # CSR capture lists (flat row-space): cap_indptr (B*N2+1), cap_indices (L,)
        # Row id is r = b*N2 + i
        # ----------------------------------------------------------------------

        # (1) Build inverse map (b, root) -> members j (opponent stones only)
        is_opp = (self.board_flatten == opp_colour)                    # (B,N2)
        b_flat  = torch.arange(B,  device=dev, dtype=IDX_DTYPE).repeat_interleave(N2)  # (B*N2,)
        j_flat  = torch.arange(N2, device=dev, dtype=IDX_DTYPE).repeat(B)              # (B*N2,)
        r_flat  = roots.reshape(-1)                                                    # (B*N2,)
        keep    = is_opp.reshape(-1)                                                   # (B*N2,) bool

        key_root = b_flat[keep] * N2 + r_flat[keep]    # (Kopp,)
        j_list   = j_flat[keep]                        # (Kopp,)

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
            root_b = (uniq // N2).to(IDX_DTYPE)     # (G,)
            root_r = (uniq %  N2).to(IDX_DTYPE)     # (G,)
            root_start = torch.full((B, N2), -1, dtype=IDX_DTYPE, device=dev)
            root_end   = torch.full((B, N2), -1, dtype=IDX_DTYPE, device=dev)
            root_start[root_b, root_r] = starts.to(IDX_DTYPE)
            root_end[root_b, root_r]   = ends.to(IDX_DTYPE)
        else:
            j_sorted   = torch.empty(0, dtype=IDX_DTYPE, device=dev)
            root_start = torch.full((B, N2), -1, dtype=IDX_DTYPE, device=dev)
            root_end   = torch.full((B, N2), -1, dtype=IDX_DTYPE, device=dev)

        # (2) For each move (b,i), find which neighbor roots would be captured (≤4),
        #     dedup per move, and get the slice lengths from the global root→members map.

        cand_roots = torch.full((B, N2, 4), SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE, device=dev)
        cand_roots[can_capture] = neigh_roots[can_capture]  # (B, N2, 4)

        # Sort per-move so duplicates sit next to each other
        roots_sorted, _ = cand_roots.sort(dim=2)           # (B, N2, 4)

        # Mark valid entries (not -1)
        valid = roots_sorted >= 0                           # (B, N2, 4)

        # Dedup within each move: keep the first of any equal run along the last axis
        same_as_prev = torch.zeros_like(valid)
        same_as_prev[..., 1:] = (roots_sorted[..., 1:] == roots_sorted[..., :-1]) & valid[..., 1:] & valid[..., :-1]
        unique_mask = valid & ~same_as_prev                 # True only for the unique roots per move

        # Gather start/end of each root’s member slice from the global maps.
        idx_flat = roots_sorted.clamp_min(0).reshape(B, -1)         # (B, N2*4)
        lo_flat  = torch.gather(root_start, 1, idx_flat)            # (B, N2*4)
        hi_flat  = torch.gather(root_end,   1, idx_flat)            # (B, N2*4)
        lo = lo_flat.view(B, N2, 4)                                  # (B, N2, 4)
        hi = hi_flat.view(B, N2, 4)                                  # (B, N2, 4)

        # Segment lengths per captured root (0 where invalid/duplicate)
        seg_len = torch.where(unique_mask & (lo >= 0) & (hi > lo), hi - lo, torch.zeros_like(lo))  # (B, N2, 4)

        # (3) Row-local counts and GLOBAL flat indptr (length R+1)
        cap_counts      = seg_len.sum(dim=2).to(torch.int32)         # (B, N2)
        cap_counts_flat = cap_counts.reshape(-1)                     # (R,)
        R = B * N2

        cap_indptr_flat = torch.empty(R + 1, dtype=torch.int32, device=dev)
        cap_indptr_flat[0] = 0
        torch.cumsum(cap_counts_flat.to(torch.int32), 0, out=cap_indptr_flat[1:])  # row-major order

        total_L = int(cap_indptr_flat[-1].item())
        cap_indices = torch.empty(total_L, dtype=torch.int32, device=dev)

        # (4) Fill cap_indices by concatenating member slices per (b,i) root — fully vectorized
        keep_trip = seg_len > 0                       # (B, N2, 4) bool
        # Build per-move identifiers
        b_ar = torch.arange(B,  device=dev, dtype=IDX_DTYPE).view(B, 1, 1).expand_as(seg_len)
        i_ar = torch.arange(N2, device=dev, dtype=IDX_DTYPE).view(1, N2, 1).expand_as(seg_len)

        b_tri   = b_ar[keep_trip]                     # (T,)
        i_tri   = i_ar[keep_trip]                     # (T,)
        lo_tri  = lo[keep_trip]                       # (T,)
        len_tri = seg_len[keep_trip].to(torch.int64)  # (T,)

        # Row id in flat space
        row_flat = (b_tri * N2 + i_tri).to(torch.int64)          # (T,)
        base_row = cap_indptr_flat[row_flat]                     # (T,) int32

        # Sort triples by row to make per-row concatenation contiguous
        sort_key, ord = torch.sort(row_flat)
        row_flat = row_flat[ord]; base_row = base_row[ord]
        lo_tri   = lo_tri[ord];   len_tri  = len_tri[ord]

        # Running offsets within each row (0, n1, n1+n2, …)
        same_prev = torch.zeros_like(sort_key, dtype=torch.bool)
        same_prev[1:] = (sort_key[1:] == sort_key[:-1])
        csum  = torch.cumsum(len_tri, dim=0)                      # (T,)
        row_offset = torch.zeros_like(csum)
        row_offset[1:] = torch.where(same_prev[1:], csum[:-1], torch.zeros_like(csum[:-1]))

        # Element-level expansion (safe even when K == 0)
        K = int(len_tri.sum().item())
        elem_pos   = torch.arange(K, device=dev, dtype=torch.int64)                                # (K,)
        off_in_run = elem_pos - torch.repeat_interleave(row_offset, len_tri)                       # (K,)

        src_idx = torch.repeat_interleave(lo_tri, len_tri) + off_in_run                            # (K,)
        dst_idx = (torch.repeat_interleave((base_row.to(torch.int64) + row_offset), len_tri)
                   + off_in_run)                                                                   # (K,)

        # Scatter-copy (no-op when K==0)
        cap_indices[dst_idx] = j_sorted[src_idx].to(torch.int32)

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
        roots     : (B,N2) int64  union-find representative per point
        root_libs : (B,N2) int64  liberty count per root id (index by root id)
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

        # Hook & compress
        parent = torch.arange(N2, dtype=IDX_DTYPE, device=dev).expand(B, N2)      # (B,N2)
        parent = self._hook_and_compress(parent, same)
        roots  = parent                                                           # (B,N2)

        # Count unique liberties per root
        is_lib     = (neigh_cols == -1) & neigh_valid_flatten_b                   # (B,N2,4)
        stone_mask = (board_flatten != -1)                                        # (B,N2)

        libs_per_root = torch.zeros(B * N2, dtype=IDX_DTYPE, device=dev)          # (B*N2,)

        batch_map = torch.arange(B, dtype=IDX_DTYPE, device=dev).view(B, 1, 1)
        roots_exp = roots.unsqueeze(2)                                            # (B,N2,1)
        lib_idx   = self.neigh_index_flatten.view(1, N2, 4)                       # (1,N2,4)
        mask      = is_lib & stone_mask.unsqueeze(2)                               # (B,N2,4)

        # edges → pairs (root_key, liberty_key) for scatter dedup
        fb = batch_map.expand_as(mask)[mask]                                       # (K,)
        fr = roots_exp.expand_as(mask)[mask]                                       # (K,)
        fl = lib_idx.expand_as(mask)[mask]                                         # (K,)

        key_root = fb * N2 + fr
        key_lib  = fb * N2 + fl
        pairs    = torch.stack((key_root, key_lib), dim=1)                         # (K,2)

        sort_key     = pairs[:, 0] * (N2 * B) + pairs[:, 1]
        sorted_idx   = sort_key.argsort()
        pairs_sorted = pairs[sorted_idx]
        uniq         = torch.unique_consecutive(pairs_sorted, dim=0)

        libs_per_root.scatter_add_(0, uniq[:, 0], torch.ones_like(uniq[:, 0], dtype=IDX_DTYPE))
        root_libs = libs_per_root.reshape(B, N2)                                   # (B,N2)

        return roots, root_libs

    # ------------------------------------------------------------------
    # Batched pointer-jumping with periodic convergence check
    # ------------------------------------------------------------------
    def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        B, N2 = parent.shape
        neigh_index_flatten_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1)
        max_rounds = N2  # upper bound; early exit will kick in

        for i in range(max_rounds):
            parent_prev = parent
            nbr_parent  = torch.gather(
                parent, 1, neigh_index_flatten_b.clamp(min=0).reshape(B, -1)
            ).view(B, N2, 4)                                                      # (B,N2,4)
            nbr_parent  = torch.where(same, nbr_parent, torch.full_like(nbr_parent, N2))
            min_nbr     = nbr_parent.min(dim=2).values                             # (B,N2)

            hooked = torch.minimum(parent, min_nbr)                                # (B,N2)
            comp   = torch.gather(hooked, 1, hooked)                               # parent[parent]
            comp   = torch.gather(comp,   1, comp)                                 # parent[parent[parent]]

            # lazy convergence check (every 4 iters)
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
        neigh_index_f_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)  # (B,N2,4)
        neigh_valid_f_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)               # (B,N2,4) bool

        board3  = self.board_flatten.unsqueeze(2).expand(-1, -1, 4)                                # (B,N2,4)
        gathered = torch.gather(board3, dim=1, index=neigh_index_f_b)                              # (B,N2,4)

        out = torch.where(neigh_valid_f_b, gathered.to(DTYPE_COLOR),
                          torch.full_like(gathered, SENTINEL_NEIGH_COLOR, dtype=DTYPE_COLOR))
        return out  # (B,N2,4)

    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        B, N2 = roots.shape
        neigh_index_f_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)  # (B,N2,4)
        neigh_valid_f_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)               # (B,N2,4) bool

        roots3  = roots.unsqueeze(2).expand(-1, -1, 4)                                            # (B,N2,4)
        gathered = torch.gather(roots3, dim=1, index=neigh_index_f_b)                             # (B,N2,4)

        out = torch.where(neigh_valid_f_b, gathered,
                          torch.full_like(gathered, SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE))
        return out  # (B,N2,4)
