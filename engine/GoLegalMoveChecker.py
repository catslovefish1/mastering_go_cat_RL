# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py – board-plane edition (CSR capture; no dense BxN2xN2)
============================================================================

Board
-----
- board: (B, H, W) int8 with values: -1 empty, 0 black, 1 white
- Internally we work on a flattened grid: N2 = H * W

CSR nomenclature used below
--------------------------
stone_global_index               : (K,)    int32  # all stone cell-ids, concatenated group-major
stone_global_pointer             : (R+1,)  int32  # CSR indptr over all groups in the batch
group_global_pointer_per_board   : (B+1,)  int32  # per-board offset of groups (local→global bridge)
stone_local_index_from_cell      : (B,N2)  int32  # (b, cell) → local gid (−1 for empty)
stone_local_index_from_root      : (B,N2)  int32  # (b, UF root) → local gid (−1 if no stones at that root)
captured_group_local_index       : (B,N2,4)int32  # per candidate cell, up to 4 capturable neighbor groups (−1 where none)
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict
from contextlib import contextmanager

import torch
from functools import wraps

    
def _mps_driver_mb(dev: Optional[torch.device]) -> float:
        if dev is None:
            return 0.0
        if dev.type == "mps" and hasattr(torch, "mps"):
            return torch.mps.driver_allocated_memory() / (1024 ** 2)
        return 0.0





# =============================================================================
# Public API
# =============================================================================

class GoLegalMoveChecker:
    def __init__(self, board_size=19, device=None):
        self._checker = VectorizedBoardChecker(board_size, device)

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

    # Per-board (flatten) structures (static for a given board size)
    index_flatten: torch.Tensor          # (N2,)
    neigh_index_flatten: torch.Tensor    # (N2,4) int64
    neigh_valid_flatten: torch.Tensor    # (N2,4) bool

    # Per-call (runtime) data
    board_flatten: torch.Tensor          # (B,N2), set each call
    
    

    def __init__(self, board_size: int, device: Optional[torch.device]):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device

        # UF workspace (lazy, depends on B)
        self._uf_nbr_parent: Optional[torch.Tensor] = None  # (B,N2,4) int32, allocated on first use
        # CSR debug state
        self._csr_debug_id = 0
        
        self._csr_capacity_K = 0
        self._csr_capacity_R = 0
        self._csr_sg = None      # stone_global_index
        self._csr_sp = None      # stone_global_pointer
        self._csr_slc = None     # stone_local_index_from_cell
        self._csr_slr = None     # stone_local_index_from_root
        self._csr_gptr = None    # group_global_pointer_per_board


        self._init_board_fallten_structure()


    # ------------------------------------------------------------------
    # Precomputed 4-neighbour tables in flat space
    # ------------------------------------------------------------------
    def _init_board_fallten_structure(self) -> None:
        N, N2, dev = self.board_size, self.N2, self.device

        # (N2,) flat indices of a single board
        self.index_flatten = torch.arange(N2, dtype=torch.int64, device=dev)

        # (N2,4) neighbours via offsets: N,S,W,E
        OFF  = torch.tensor([-N, N, -1, 1], dtype=torch.int64, device=dev)    # (4,)
        nbrs = self.index_flatten[:, None] + OFF                              # (N2,4)

        # Edge handling
        valid = (nbrs >= 0) & (nbrs < N2)                                     # (N2,4)
        col   = self.index_flatten % N
        valid[:, 2] &= col != 0           # W invalid at left edge
        valid[:, 3] &= col != N - 1       # E invalid at right edge

        self.neigh_index_flatten = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.neigh_valid_flatten = valid

        # Non-negative neighbor indices (off-board → 0) for gather
        self.neigh_index_nonneg_flatten = torch.where(
            self.neigh_valid_flatten,
            self.neigh_index_flatten,
            torch.zeros_like(self.neigh_index_flatten),
        )  # (N2,4) int64


    # ------------------------------------------------------------------
    # Top-level: legal mask + capture metadata (CSR-based; no dense BxN2xN2)
    # ------------------------------------------------------------------
    def compute_batch_legal_and_info(
        self,
        board: torch.Tensor,          # (B,H,W) int8 in {-1,0,1}
        current_player: torch.Tensor  # (B,)    uint8 in {0,1}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, H, W = board.shape
        N2      = self.N2

        # Per-call runtime flatten (depends on B)
        self.board_flatten = board.reshape(B, N2)         # (B,N2)
        empty              = (self.board_flatten == -1)   # (B,N2) bool

        # ---- Neighbour colors: compute ONCE per call and reuse ----
        neigh_colors = self._get_neighbor_colors_batch()  # (B,N2,4)

        # Groups (roots) + liberties (reuse neigh_colors inside)
        roots, root_libs = self._batch_init_union_find(neigh_colors)  # (B,N2) each

        # === Build the CSR + LUTs (batch-wide) ==================================
        csr = self._build_group_csr(roots)
        stone_global_index              = csr["stone_global_index"]               # (K,)
        stone_global_pointer            = csr["stone_global_pointer"]             # (R+1,)
        group_global_pointer_per_board  = csr["group_global_pointer_per_board"]   # (B+1,)
        stone_local_index_from_cell     = csr["stone_local_index_from_cell"]      # (B,N2)
        stone_local_index_from_root     = csr["stone_local_index_from_root"]      # (B,N2)

        # === Neighbour tables ====================================================
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)
        neigh_roots  = self._get_neighbor_roots_batch(roots)         # (B,N2,4)

        curr = current_player.view(B, 1, 1)  # (B,1,1)
        opp  = 1 - curr

        # A) immediate liberties
        has_any_lib = ((neigh_colors == -1) & neigh_valid_flatten_b).any(dim=2)   # (B,N2)

        # B) captures: adjacent opponent group with exactly 1 liberty
        neigh_roots_f = neigh_roots.reshape(B, -1)                                # (B,N2*4)
        neigh_libs_f  = root_libs.gather(1, neigh_roots_f.clamp(min=0))           # (B,N2*4)
        neigh_libs    = neigh_libs_f.view(B, N2, 4)                                # (B,N2,4)

        opp_mask        = (neigh_colors == opp) & neigh_valid_flatten_b           # (B,N2,4) bool
        can_capture     = opp_mask & (neigh_libs == 1)                            # (B,N2,4)
        can_capture_any = can_capture.any(dim=2)                                  # (B,N2)

        # C) friendly safe attachment
        friendly     = (neigh_colors == curr) & neigh_valid_flatten_b             # (B,N2,4)
        friendly_any = (friendly & (neigh_libs > 1)).any(dim=2)                   # (B,N2)

        # D) final legality
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)                                     # (B,H,W)

        # ----- Capture metadata (CSR-based; no dense mask) -----------------------
        # Map neighbour *roots* → local group ids (or -1); keep only capturing dirs
        captured_group_local_index_all = stone_local_index_from_root.gather(
            1, neigh_roots.clamp(min=0).reshape(B, -1).to(torch.int64)
        ).view(B, N2, 4)                                                          # (B,N2,4) int32

        captured_group_local_index = torch.where(
            can_capture,
            captured_group_local_index_all,
            torch.full_like(captured_group_local_index_all, -1, dtype=torch.int32),
        )                                                                         # (B,N2,4) int32

        # info payload (no (B,N2,N2) tensors; use CSR + per-candidate gids)
        info: Dict[str, torch.Tensor] = {
            # core group topology
            "roots": roots,                          # (B,N2) int32
            "root_libs": root_libs,                  # (B,N2) int32

            # legality helpers
            "can_capture_any": can_capture_any,      # (B,N2) bool
            "captured_group_local_index": captured_group_local_index.long(),  # (B,N2,4) int32

            # CSR (global, batch-wide)
            "stone_global_index":             stone_global_index.long(),              # (K,)   int32
            "stone_global_pointer":           stone_global_pointer.long(),            # (R+1,) int32
            "group_global_pointer_per_board": group_global_pointer_per_board.long(),  # (B+1,) int32

            # LUTs (fast ID→ID maps)
            "stone_local_index_from_cell":    stone_local_index_from_cell.long(),     # (B,N2) int32
            "stone_local_index_from_root":    stone_local_index_from_root.long(),     # (B,N2) int32
        }

        return legal_mask, info


       # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # UF + liberties with reusable (B,N2,4) workspace
    # ------------------------------------------------------------------
    def _batch_init_union_find(
        self,
        neigh_cols: torch.Tensor,   # (B,N2,4) int8 – precomputed neighbour colors
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        roots     : (B,N2) int32  union-find representative per point
        root_libs : (B,N2) int32  liberty count per root id (index by root id)
        """
        board_flatten = self.board_flatten
        B, N2 = board_flatten.shape
        dev   = self.device

        # ---- DEBUG: measure UF/lib step on MPS --------------------------------
        is_mps = (dev is not None and dev.type == "mps" and hasattr(torch, "mps"))
        if is_mps:
            drv_before_uf = _mps_driver_mb(dev)
            print(f"[MPS][UF] before={drv_before_uf:.1f} MB")

        # ---- Neighbour validity mask (view only) -------------------------------
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)

        # ---- Same-color adjacency (ignore empties; respect edges) --------------
        same = (neigh_cols == board_flatten.unsqueeze(2)) \
            & (board_flatten.unsqueeze(2) != -1) \
            & neigh_valid_flatten_b                                              # (B,N2,4)

        # ---- Hook & compress (union-find) --------------------------------------
        parent = torch.arange(N2, dtype=torch.int32, device=dev).expand(B, N2)    # (B,N2)
        parent = self._hook_and_compress(parent, same)
        roots  = parent                                                           # (B,N2)

        # ---- Count unique liberties per root -----------------------------------
        is_lib     = (neigh_cols == -1) & neigh_valid_flatten_b                   # (B,N2,4)
        stone_mask = (board_flatten != -1)                                        # (B,N2)
        mask       = is_lib & stone_mask.unsqueeze(2)                             # (B,N2,4)

        # K = number of stone→empty edges across the batch
        fb, fj, fd = torch.where(mask)            # each (K,)
        fr = roots[fb, fj]                        # (K,)  root id of the stone
        fl = self.neigh_index_flatten[fj, fd]     # (K,) liberty cell id (valid)

        # Deduplicate by (batch, root, liberty_point) then count uniques per root
        key_root = fb * N2 + fr                   # (K,)
        key_lib  = fb * N2 + fl                   # (K,)
        pairs    = torch.stack((key_root, key_lib), dim=1)        # (K,2)

        sort_key     = pairs[:, 0].to(torch.int64) * (N2 * B) + pairs[:, 1].to(torch.int64)
        sorted_idx   = sort_key.argsort()
        pairs_sorted = pairs[sorted_idx]
        uniq         = torch.unique_consecutive(pairs_sorted, dim=0)              # (Kuniq,2)

        libs_per_root = torch.zeros(B * N2, dtype=torch.int32, device=dev)        # (B*N2,)
        if uniq.numel() > 0:
            libs_per_root.scatter_add_(
                0,
                uniq[:, 0],
                torch.ones(uniq.size(0), dtype=torch.int32, device=dev)
            )

        root_libs = libs_per_root.view(B, N2)                                     # (B,N2)

        # ---- DEBUG: print K and driver delta -----------------------------------
        if is_mps:
            K_edges = int(mask.sum().item())          # total stone→liberty edges
            Kuniq   = int(uniq.size(0))               # unique (root, liberty) pairs
            drv_after_uf = _mps_driver_mb(dev)
            print(
                f"[MPS][UF]  after={drv_after_uf:.1f} MB "
                f"Δ={drv_after_uf - drv_before_uf:+.1f} MB "
                f"K_edges={K_edges} Kuniq={Kuniq}"
            )

        return roots, root_libs



    # ------------------------------------------------------------------
    # UF workspace helper + batched pointer-jumping (int32)
    # ------------------------------------------------------------------
    def _ensure_uf_workspace(self, B: int, N2: int, dev: torch.device) -> torch.Tensor:
        """
        Ensure we have a reusable (B,N2,4) int32 workspace for UF neighbor parents.
        We treat the buffer as a *capacity* buffer:
        - allocate once with at least (B,N2,4)
        - reuse it forever
        - if we ever need larger B or N2, grow it, but never shrink.
        """
        ws = self._uf_nbr_parent

        # Determine required shape
        req_B  = B
        req_N2 = N2
        req_shape = (req_B, req_N2, 4)

        # If no buffer yet, or on a different device / dtype, create fresh
        if ws is None or ws.device != dev or ws.dtype != torch.int32:
            ws = torch.empty(req_shape, dtype=torch.int32, device=dev)
            self._uf_nbr_parent = ws
            return ws

        # If existing buffer is too small in B or N2, grow capacity
        cur_B, cur_N2, cur_4 = ws.shape
        if cur_B < req_B or cur_N2 < req_N2 or cur_4 != 4:
            new_B  = max(cur_B, req_B)
            new_N2 = max(cur_N2, req_N2)
            ws = torch.empty((new_B, new_N2, 4), dtype=torch.int32, device=dev)
            self._uf_nbr_parent = ws
            return ws[:req_B, :req_N2]

        # Otherwise, reuse existing buffer; just slice to requested view
        return ws[:req_B, :req_N2]



    def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        """
        Batched union–find with pointer jumping.

        parent : (B,N2) int32
        same   : (B,N2,4) bool – adjacency for N,S,W,E
        """
        B, N2 = parent.shape
        dev   = parent.device   # <- use the real tensor device, not self.device

        # Precomputed neighbor indices (flat), clamped off-board to 0
        nbr_idx = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)

        # Reusable (B,N2,4) int32 workspace (capacity-based, sliced)
        nbr_parent_ws = self._ensure_uf_workspace(B, N2, dev)

        # --- DEBUG (optional) ---------------------------------------------------
        is_mps = (dev.type == "mps" and hasattr(torch, "mps"))
        if is_mps:
            drv_before = _mps_driver_mb(dev)
            print(
                f"[MPS][UFhook] parent_ptr={parent.data_ptr()} "
                f"ws_ptr={nbr_parent_ws.data_ptr()} "
                f"nbr_idx_ptr={nbr_idx.data_ptr()}"
            )

        max_rounds = (N2).bit_length() + 10

        for i in range(max_rounds):
            parent3 = parent.view(B, N2, 1).expand(-1, -1, 4)  # view, no alloc

            torch.take_along_dim(
                parent3,
                nbr_idx,
                dim=1,
                out=nbr_parent_ws,
            )

            nbr_parent_ws.masked_fill_(~same, N2)

            min_nbr = nbr_parent_ws.min(dim=2).values
            hooked  = torch.minimum(parent, min_nbr)
            parent_next = torch.gather(hooked, 1, hooked.long())

            if (i & 3) == 3 and torch.equal(parent_next, parent):
                parent = parent_next
                break

            parent = parent_next

        if is_mps:
            drv_after = _mps_driver_mb(dev)
            print(
                f"[MPS][UFhook] full-iter driver={drv_before:.1f}->{drv_after:.1f} MB "
                f"Δ={drv_after - drv_before:+.1f} MB "
                f"parent_final_ptr={parent.data_ptr()} "
                f"ws_ptr={nbr_parent_ws.data_ptr()}"
            )

        return parent



       # ------------------------------------------------------------------
    # Build CSR + LUTs (global across batch; safe for K=0)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_group_csr(self, roots: torch.Tensor):
        dev = self.device or torch.device("cpu")
        B, N2 = self.board_flatten.shape

        # ---- fixed capacity (worst case for this (B, N2)) -----------------
        maxK = B * N2           # worst-case stones
        maxR = B * N2           # worst-case groups (overkill but simple)

        # Grow CSR workspaces if needed (only on first use / if B, N2 changed)
        if self._csr_capacity_K < maxK:
            self._csr_sg  = torch.empty(maxK, dtype=torch.int32, device=dev)
            self._csr_slc = torch.full((B, N2), -1, dtype=torch.int32, device=dev)
            self._csr_slr = torch.full((B, N2), -1, dtype=torch.int32, device=dev)
            self._csr_capacity_K = maxK

        if self._csr_capacity_R < maxR:
            self._csr_sp   = torch.empty(maxR + 1, dtype=torch.int32, device=dev)
            self._csr_gptr = torch.empty(B + 1, dtype=torch.int32, device=dev)
            self._csr_capacity_R = maxR

        # --- debug bookkeeping ----------------------------------------
        self._csr_debug_id += 1
        csr_id = self._csr_debug_id
        is_mps = (dev.type == "mps" and hasattr(torch, "mps"))

        drv_before = _mps_driver_mb(dev) if is_mps else 0.0
        if is_mps:
            print(f"[MPS][csr:{csr_id}] before_csr={drv_before:.1f} MB")

        # 1) take stones (exclude empties)
        is_stone = (self.board_flatten != -1)                    # (B,N2)
        b_all, j_all = is_stone.nonzero(as_tuple=True)           # (K,), (K,)
        r_all = roots[b_all, j_all]                              # (K,)

        # 2) global sort by (board, root) → stones contiguous per group
        sort_key = b_all * (N2 + 1) + r_all
        perm     = sort_key.argsort()
        b_sorted = b_all[perm]                                   # (K,)
        j_sorted = j_all[perm]                                   # (K,)
        r_sorted = r_all[perm]                                   # (K,)

        K = b_sorted.numel()

        # 3) run boundaries for (board, root) (no special-case at 0)
        same_prev = (b_sorted == torch.roll(b_sorted, 1)) & (r_sorted == torch.roll(r_sorted, 1))
        new_group = (~same_prev) | (torch.arange(K, device=dev) == 0)     # (K,) bool

        run_starts = torch.nonzero(new_group, as_tuple=True)[0]           # (R,)
        R          = run_starts.numel()
        run_board  = b_sorted[run_starts]                                 # (R,)
        run_idx    = torch.arange(R, device=dev, dtype=torch.int64)       # (R,)

        run_id_for_stone = new_group.to(torch.int64).cumsum(0) - 1        # (K,)
        run_sizes        = torch.bincount(
            run_id_for_stone.clamp_min(0), minlength=R
        ).to(torch.int32)                                                 # (R,)

        # 4) board pointers (local→global bridge)
        groups_per_board = torch.bincount(
            run_board.to(torch.int64), minlength=B
        ).to(torch.int32)                                                 # (B,)

        # ---- USE PREALLOCATED BUFFERS HERE -----------------------------
        # group_global_pointer_per_board: prefix of _csr_gptr
        group_global_pointer_per_board = self._csr_gptr[:B+1]
        group_global_pointer_per_board.zero_()
        group_global_pointer_per_board[1:] = groups_per_board.cumsum(0)
        board_first_global = group_global_pointer_per_board[:-1].to(torch.int64)  # (B,)

        # local gid per run
        gid_of_run_local = (
            run_idx - board_first_global.index_select(0, run_board)
        ).to(torch.int32)                                                 # (R,)
        gid_for_stone_local = gid_of_run_local[run_id_for_stone]          # (K,)

        # 5) outputs (CSR arrays + LUTs) – all via slices of workspaces
        # stone_global_index: prefix of _csr_sg
        stone_global_index = self._csr_sg[:K]
        if K > 0:
            stone_global_index.copy_(j_sorted.to(torch.int32))

        # stone_global_pointer: prefix of _csr_sp
        stone_global_pointer = self._csr_sp[:R+1]
        stone_global_pointer.zero_()
        if R > 0:
            stone_global_pointer[1:R+1] = run_sizes.cumsum(0)

        # (b, cell) -> local gid  (scatter on flat buffer)
        stone_local_index_from_cell = self._csr_slc   # (B,N2)
        stone_local_index_from_cell.fill_(-1)
        if K > 0:
            lin_cells = (b_sorted * N2 + j_sorted).to(torch.int64)        # (K,)
            stone_local_index_from_cell.view(-1).index_put_(
                (lin_cells,), gid_for_stone_local, accumulate=False
            )

        # (b, UF root) -> local gid
        stone_local_index_from_root = self._csr_slr   # (B,N2)
        stone_local_index_from_root.fill_(-1)
        if R > 0:
            root_id_of_run = r_sorted[run_starts]                         # (R,)
            lin_roots = (run_board * N2 + root_id_of_run).to(torch.int64) # (R,)
            stone_local_index_from_root.view(-1).index_put_(
                (lin_roots,), gid_of_run_local, accumulate=False
            )

        # --- CSR memory debug summary ---------------------------------
        if is_mps:
            drv_after = _mps_driver_mb(dev)
            delta_drv = drv_after - drv_before

            csr_bytes = (
                stone_global_index.numel()          * stone_global_index.element_size()
                + stone_global_pointer.numel()      * stone_global_pointer.element_size()
                + group_global_pointer_per_board.numel() * group_global_pointer_per_board.element_size()
                + stone_local_index_from_cell.numel()    * stone_local_index_from_cell.element_size()
                + stone_local_index_from_root.numel()    * stone_local_index_from_root.element_size()
            )
            csr_mb = csr_bytes / (1024 ** 2)

            ptr_sg = stone_global_index.data_ptr()
            ptr_sp = stone_global_pointer.data_ptr()
            ptr_gc = group_global_pointer_per_board.data_ptr()

            print(
                f"[MPS][csr:{csr_id}] after_csr={drv_after:.1f} MB "
                f"Δ_drv={delta_drv:+.1f} MB "
                f"csr≈{csr_mb:.1f} MB "
                f"K={K} R={R} "
                f"ptr_sg={ptr_sg} ptr_sp={ptr_sp} ptr_gc={ptr_gc}"
            )

        return {
            "stone_global_index":              stone_global_index,              # (K,)
            "stone_global_pointer":            stone_global_pointer,            # (R+1,)
            "group_global_pointer_per_board":  group_global_pointer_per_board,  # (B+1,)
            "stone_local_index_from_cell":     stone_local_index_from_cell,     # (B,N2)
            "stone_local_index_from_root":     stone_local_index_from_root,     # (B,N2)
        }



    # ------------------------------------------------------------------
    # Neighbour helpers (batched, flat graph, 4 dirs)
    # ------------------------------------------------------------------
    def _get_neighbor_colors_batch(self) -> torch.Tensor:
        """Return neighbor colors pulled from self.board_flatten without per-call index clamps."""
        B, N2 = self.board_flatten.shape

        # Views only (no alloc): expand precomputed indices & validity
        idx   = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)  # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)         # (B,N2,4) bool (view)

        # Gather from a broadcasted view of the board (no data copy)
        board3 = self.board_flatten.unsqueeze(2).expand(-1, -1, 4)                # (B,N2,4) view
        out = torch.gather(board3, dim=1, index=idx).to(torch.int8)               # (B,N2,4) int8

        # Mark off-board neighbors distinctly
        out.masked_fill_(~valid, -2)
        return out

    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        """Return neighbor union-find roots using precomputed non-negative indices; off-board = -1."""
        B, N2 = roots.shape

        # Views only (no alloc)
        idx   = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)  # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)         # (B,N2,4) bool (view)

        roots3   = roots.unsqueeze(2).expand(-1, -1, 4)                            # (B,N2,4) view
        gathered = torch.gather(roots3, dim=1, index=idx)                          # (B,N2,4)
        gathered.masked_fill_(~valid, -1)
        return gathered
