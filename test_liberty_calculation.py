import torch
import numpy as np
from engine import GoLegalMoveChecker as legal_module
from engine.GoLegalMoveChecker import VectorizedBoardChecker
GoLegalMoveChecker = legal_module.GoLegalMoveChecker

def test_liberty_calculation():
    """Detailed analysis of liberty calculation for Root 0."""
    
    board_size = 5
    checker = GoLegalMoveChecker(board_size=board_size)
    
    print("="*80)
    print("DETAILED LIBERTY CALCULATION FOR ROOT 0 (WHITE GROUP)")
    print("="*80)
    
    # Create the specific board position
    board_flat = torch.tensor([
        1,   # [0,0] white
        -1,  # [0,1] empty
        0,   # [0,2] black
        0,   # [0,3] black
        0,   # [0,4] black
        1,   # [1,0] white
        1,   # [1,1] white
        0,   # [1,2] black
        0,   # [1,3] black
        0,   # [1,4] black
        1,   # [2,0] white
        0,   # [2,1] black
        -1,  # [2,2] empty
        0,   # [2,3] black
        0,   # [2,4] black
        1,   # [3,0] white
        1,   # [3,1] white
        0,   # [3,2] black
        0,   # [3,3] black
        0,   # [3,4] black
        1,   # [4,0] white
        1,   # [4,1] white
        1,   # [4,2] white
        0,   # [4,3] black
        0    # [4,4] black
    ], dtype=torch.int8)
    
    board = board_flat.view(1, board_size, board_size)
    
    print("\n--- BOARD STATE ---")
    print("Board (1=white, 0=black, -1=empty):")
    board_np = board[0].numpy()
    for i, row in enumerate(board_np):
        row_str = ' '.join(f"{cell:2d}" for cell in row)
        print(f"Row {i}: {row_str}")
    
    # Get internal checker
    internal_checker = checker._checker
    N2 = board_size * board_size
    
    # Flatten board
    board_f = board.view(1, -1)
    
    # Get union-find results - this is where liberty calculation happens
    _parent, colour, roots, root_libs = internal_checker._batch_init_union_find(board_f)
    
    print("\n--- UNION-FIND GROUPS ---")
    roots_2d = roots[0].view(board_size, board_size).numpy()
    for i in range(board_size):
        print(f"Row {i}: {' '.join(f'{roots_2d[i,j]:3d}' for j in range(board_size))}")
    
    # Focus on Root 0 (white group)
    root_0_positions = []
    for idx in range(N2):
        if roots[0, idx] == 0:
            row, col = idx // board_size, idx % board_size
            root_0_positions.append((row, col, idx))
    
    print(f"\n--- ROOT 0 (WHITE GROUP) ANALYSIS ---")
    print(f"Total stones in group: {len(root_0_positions)}")
    print("Positions in group:")
    for row, col, idx in root_0_positions:
        print(f"  [{row},{col}] (flat index {idx})")
    
    # Now let's manually trace the liberty calculation
    print("\n--- MANUAL LIBERTY CALCULATION ---")
    print("Checking neighbors of each stone in Root 0 for empty spaces:")
    
    # Get neighbor indices
    neigh_idx = internal_checker.NEIGH_IDX  # (N2, 4)
    neigh_valid = internal_checker.NEIGH_VALID  # (N2, 4)
    
    directions = ['North', 'South', 'West', 'East']
    liberties_found = set()
    
    for row, col, idx in root_0_positions:
        print(f"\n  Stone at [{row},{col}] (index {idx}):")
        for d in range(4):
            neighbor_idx = neigh_idx[idx, d].item()
            is_valid = neigh_valid[idx, d].item()
            
            if is_valid and neighbor_idx >= 0:
                n_row, n_col = neighbor_idx // board_size, neighbor_idx % board_size
                n_value = board_f[0, neighbor_idx].item()
                
                if n_value == -1:  # Empty
                    liberties_found.add(neighbor_idx)
                    print(f"    {directions[d]:5s}: [{n_row},{n_col}] = EMPTY (liberty!)")
                elif n_value == 1:
                    print(f"    {directions[d]:5s}: [{n_row},{n_col}] = white")
                else:
                    print(f"    {directions[d]:5s}: [{n_row},{n_col}] = black")
            elif not is_valid:
                print(f"    {directions[d]:5s}: out of bounds")
    
    print(f"\n--- LIBERTY SUMMARY ---")
    print(f"Unique liberty positions found: {len(liberties_found)}")
    for lib_idx in liberties_found:
        lib_row, lib_col = lib_idx // board_size, lib_idx % board_size
        print(f"  Liberty at [{lib_row},{lib_col}] (index {lib_idx})")
    
    print(f"\nAlgorithm reports: {root_libs[0, 0].item()} liberties for Root 0")
    print(f"Manual count: {len(liberties_found)} liberties")
    
    # Let's also check what the algorithm is actually doing
    print("\n--- ALGORITHM'S LIBERTY CALCULATION INTERNALS ---")
    
    # Recreate the liberty counting logic from _batch_init_union_find
    neigh_cols = internal_checker._get_neighbor_colors_batch(colour)
    is_lib = (neigh_cols == -1) & internal_checker.NEIGH_VALID.view(1, N2, 4)
    stone_mask = (colour != -1)
    
    print("\nFor each position in Root 0, checking which neighbors are liberties:")
    for row, col, idx in root_0_positions:
        libs_at_pos = []
        for d in range(4):
            if is_lib[0, idx, d]:
                neighbor_idx = neigh_idx[idx, d].item()
                if neighbor_idx >= 0:
                    n_row, n_col = neighbor_idx // board_size, neighbor_idx % board_size
                    libs_at_pos.append(f"[{n_row},{n_col}]")
        if libs_at_pos:
            print(f"  [{row},{col}]: liberties = {', '.join(libs_at_pos)}")
    
    # Check the actual liberty calculation in the algorithm
    print("\n--- CHECKING ALGORITHM'S EXACT CALCULATION ---")
    
    # The algorithm creates (root, liberty) pairs and deduplicates
    lib_idx_tensor = internal_checker.NEIGH_IDX.view(1, N2, 4)
    mask = is_lib & stone_mask.unsqueeze(2)
    
    # For Root 0 specifically
    root_0_libs = []
    for idx in range(N2):
        if roots[0, idx] == 0:  # Part of Root 0
            for d in range(4):
                if mask[0, idx, d]:
                    lib_pos = lib_idx_tensor[0, idx, d].item()
                    root_0_libs.append(lib_pos)
    
    unique_libs = list(set(root_0_libs))
    print(f"Root 0 liberty positions (before dedup): {root_0_libs}")
    print(f"Root 0 unique liberties: {unique_libs}")
    print(f"Count: {len(unique_libs)}")
    
    for lib_idx in unique_libs:
        lib_row, lib_col = lib_idx // board_size, lib_idx % board_size
        print(f"  Liberty: [{lib_row},{lib_col}]")

if __name__ == "__main__":
    test_liberty_calculation()
