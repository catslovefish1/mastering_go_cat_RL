# debug_edge_black_00.py
# Detailed trace for GoLegalMoveChecker when BLACK plays at (0,0)
# Works with any square board (flat list, row-major; -1 empty, 0 black, 1 white)

import math
import torch

from engine import GoLegalMoveChecker as legal_module
from engine.GoLegalMoveChecker import VectorizedBoardChecker

GoLegalMoveChecker = legal_module.GoLegalMoveChecker

def debug_black_at_00(flat_list):
    """
    Prints a detailed trace of GoLegalMoveChecker's logic for playing BLACK at (0,0)
    on the board given by `flat_list` (row-major; values in {-1, 0, 1}).
    """
    # ---- Infer board size ----
    n = int(math.isqrt(len(flat_list)))
    assert n * n == len(flat_list), f"Flat list length {len(flat_list)} is not a perfect square"
    N2 = n * n

    # Build tensors
    flat  = torch.tensor(flat_list, dtype=torch.int8).clone()
    board = flat.view(1, n, n)                      # (1, n, n)
    current_player = torch.tensor([0], dtype=torch.uint8)  # BLACK to play
    curr_val = int(current_player.item())           # 0
    opp_val  = 1 - curr_val                         # 1

    checker  = GoLegalMoveChecker(board_size=n)
    internal = checker._checker  # VectorizedBoardChecker

    print("=" * 80)
    print(f"DEBUG: BLACK plays at (0,0) on a {n}x{n} board")
    print("=" * 80)

    # ---- Print board ----
    print("\nBOARD (1=white, 0=black, -1=empty):")
    for i in range(n):
        row = board[0, i].tolist()
        tag = "  <- (0,0) target" if i == 0 else ""
        print(f"Row {i}: " + " ".join(f"{v:2d}" for v in row) + tag)

    # ---- Core precomputation (like checker) ----
    board_f = board.view(1, -1)             # (1, N2)
    empty   = (board_f == -1)               # (1, N2)

    parent, colour, roots, root_libs = internal._batch_init_union_find(board_f)

    print("\nGROUP ROOTS (union-find, flat indices shown as root ids):")
    roots_2d = roots[0].view(n, n).cpu().numpy()
    for i in range(n):
        print(" ".join(f"{roots_2d[i,j]:3d}" for j in range(n)))

    print("\nLIBERTIES PER ROOT (only stones):")
    unique_roots = torch.unique(roots[0][roots[0] >= 0])
    for rid_t in unique_roots:
        rid = int(rid_t.item())
        lib = int(root_libs[0, rid].item())
        col = int(colour[0, rid].item())
        kind = "white" if col == 1 else ("black" if col == 0 else "empty")
        count = int((roots[0] == rid).sum().item())
        print(f"  root {rid:3d}: libs={lib:2d} color={kind:5s} stones={count}")

    # ---- Neighbors around (0,0) ----
    flat_idx = 0  # (0,0)
    neigh_colors = internal._get_neighbor_colors_batch(colour)  # (1,N2,4)
    neigh_roots  = internal._get_neighbor_roots_batch(roots)    # (1,N2,4)
    valid_mask   = internal.NEIGH_VALID.view(1, N2, 4)

    dirs = ["North", "South", "West", "East"]
    print("\nNEIGHBORS of (0,0):")
    for d, name in enumerate(dirs):
        v = bool(valid_mask[0, flat_idx, d].item())
        ncol = int(neigh_colors[0, flat_idx, d].item())
        nroot = int(neigh_roots[0, flat_idx, d].item())
        cstr = "white" if ncol == 1 else ("black" if ncol == 0 else ("empty" if ncol == -1 else "OUT"))
        print(f"  {name:5s}: valid={v} color={ncol:2d} ({cstr:5s}) root={nroot:3d}")

    # ---- Legality components at (0,0) for BLACK ----
    # (a) Adjacent empty?
    has_lib = ((neigh_colors[0, flat_idx] == -1) & valid_mask[0, flat_idx]).any().item()

    # (b) Capture check: neighbor opponent with exactly 1 liberty
    neigh_libs_f = root_libs.gather(1, neigh_roots.reshape(1, -1).clamp(min=0))
    neigh_libs   = neigh_libs_f.view(1, N2, 4)  # (1,N2,4)

    opp_mask = (neigh_colors[0, flat_idx] == opp_val) & valid_mask[0, flat_idx]  # (4,)
    can_capture_vec = opp_mask & (neigh_libs[0, flat_idx] == 1)                   # (4,)
    can_capture_any = bool(can_capture_vec.any().item())

    # (c) Friendly safe attach (>1 libs)
    friendly      = (neigh_colors[0, flat_idx] == curr_val) & valid_mask[0, flat_idx]  # (4,)
    friendly_safe = friendly & (neigh_libs[0, flat_idx] > 1)                            # (4,)
    friendly_any  = bool(friendly_safe.any().item())

    print("\nLEGALITY COMPONENTS for BLACK at (0,0):")
    print(f"  empty?              {bool(empty[0, flat_idx].item())}")
    print(f"  has adjacent empty? {bool(has_lib)}")
    print(f"  can capture?        {can_capture_any}")
    if can_capture_any:
        dirs_idx = torch.nonzero(can_capture_vec, as_tuple=False).view(-1).tolist()
        cap_dirs = [dirs[i] for i in dirs_idx]
        print(f"    would capture: {cap_dirs}")
    print(f"  safe friendly?      {friendly_any}")

    final_legal = bool(empty[0, flat_idx].item() and (has_lib or can_capture_any or friendly_any))
    print(f"\nFINAL: LEGAL = {final_legal}")

    # ---- Show capture_groups/total_captures from the public API ----
    legal_mask, cap_info = checker.compute_legal_moves_with_captures(
        board, current_player, return_capture_info=True
    )
    cap_groups = cap_info["capture_groups"]  # (1,H,W,4)
    total_caps = cap_info["total_captures"]  # (1,H,W)
    would_cap  = cap_info["would_capture"]   # (1,H,W)

    print("\nFROM GoLegalMoveChecker (returned capture info):")
    print(f"  legal_mask[0,0]:      {bool(legal_mask[0,0,0].item())}")
    print(f"  would_capture[0,0]:   {bool(would_cap[0,0,0].item())}")
    print(f"  total_captures[0,0]:  {int(total_caps[0,0,0].item())}")
    print("  capture_groups[0,0]:  (roots by direction N,S,W,E; -1 means none)")
    cg = cap_groups[0,0,0].cpu().tolist()
    print(f"    {cg}")

    # Sanity: show membership coordinates for any captured roots
    if any(x >= 0 for x in cg):
        roots_flat = roots[0]  # (N2,)
        print("  Captured root(s) membership (coordinates):")
        for d, rid in enumerate(cg):
            if rid >= 0:
                coords = [(i // n, i % n) for i in range(N2) if int(roots_flat[i].item()) == rid]
                print(f"    dir {dirs[d]:5s} root={rid:3d}: {coords}")

if __name__ == "__main__":
    # Example: your 7x7 edge case (row-major)
    flat_list = [
        -1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
    ]
    debug_black_at_00(flat_list)
