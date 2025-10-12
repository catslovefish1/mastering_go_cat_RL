#!/usr/bin/env python3
"""Test that the new capture_stone_mask field works correctly."""

import torch
from engine.GoLegalMoveChecker import GoLegalMoveChecker
from engine.tensor_native_stable_dense import TensorBoard, Stone

def test_capture_stone_mask():
    """Test that capture_stone_mask correctly identifies which stones would be captured."""
    
    # Create a simple 3x3 board with a capturable white stone at (1,0)
    board_flat = [
        0,  0, -1,   # Black, Black, Empty
        1,  0,  1,   # White, Black, White  
       -1,  1,  1,   # Empty, White, White
    ]
    
    # Build board
    tb = TensorBoard(batch_size=1, board_size=3, enable_super_ko=True, enable_timing=False)
    H = W = 3
    b = torch.tensor(board_flat, dtype=torch.int8, device=tb.device).view(H, W)
    tb.board[0] = b
    tb.current_player[0] = 0  # Black to play
    tb._invalidate_cache()
    
    # Get legal moves and capture info
    legal = tb.legal_moves()
    cap_info = tb._last_capture_info
    
    # Check capture_stone_mask
    if "capture_stone_mask" in cap_info:
        capture_stone_mask = cap_info["capture_stone_mask"]  # (B, N2, N2)
        
        # Check move at (2,0) which should capture the white stone at (1,0)
        move_idx = 2 * W + 0  # Linear index for (2,0)
        captured_idx = 1 * W + 0  # Linear index for (1,0)
        
        print("Testing capture_stone_mask:")
        print(f"  Move at (2,0) linear index: {move_idx}")
        print(f"  White stone at (1,0) linear index: {captured_idx}")
        
        # Check if the mask correctly identifies the capture
        captures_white = capture_stone_mask[0, move_idx, captured_idx].item()
        print(f"  Does (2,0) capture (1,0)? {captures_white}")
        
        # Show all stones that would be captured by playing at (2,0)
        captured_stones = capture_stone_mask[0, move_idx].nonzero(as_tuple=True)[0]
        print(f"  All stones captured by (2,0): {captured_stones.tolist()}")
        
        # The mask should only show position 3 (the white stone at (1,0))
        assert captures_white, "Should capture the white stone at (1,0)"
        assert captured_stones.tolist() == [captured_idx], f"Should only capture stone at index {captured_idx}, but got {captured_stones.tolist()}"
        
        print("\n✓ capture_stone_mask correctly identifies captured stones!")
        print("✓ It only includes opponent stones that would actually be captured")
        
    else:
        print("ERROR: capture_stone_mask not found in capture_info")
        print("Available keys:", list(cap_info.keys()))

if __name__ == "__main__":
    test_capture_stone_mask()
