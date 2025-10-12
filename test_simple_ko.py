"""
Simple test for super-ko implementation.
"""

import torch
from engine.tensor_native_stable import TensorBoard
import time

def test_basic_super_ko():
    """Test basic super-ko functionality."""
    print("Testing super-ko on 3x3 board...")
    
    # Create a small board for faster testing
    board = TensorBoard(batch_size=1, board_size=3, enable_super_ko=True)
    
    # Make a few moves
    moves = [
        [0, 0],  # Black
        [0, 1],  # White
        [1, 0],  # Black
        [1, 1],  # White
    ]
    
    start = time.time()
    for i, move in enumerate(moves):
        print(f"Move {i+1}: {'Black' if i % 2 == 0 else 'White'} at {move}")
        board.step(torch.tensor([move]))
    
    # Check legal moves (this is where super-ko check happens)
    legal = board.legal_moves()
    elapsed = time.time() - start
    
    print(f"\nBoard state:")
    print(board.board[0])
    print(f"\nLegal moves: {legal.sum().item()}")
    print(f"Time taken: {elapsed:.3f}s")
    
    if elapsed < 1.0:
        print("✅ Performance is acceptable!")
    else:
        print("⚠️ Performance issue detected - too slow!")

def test_without_super_ko():
    """Test without super-ko for comparison."""
    print("\n\nTesting WITHOUT super-ko on 3x3 board...")
    
    board = TensorBoard(batch_size=1, board_size=3, enable_super_ko=False)
    
    moves = [
        [0, 0],  # Black
        [0, 1],  # White  
        [1, 0],  # Black
        [1, 1],  # White
    ]
    
    start = time.time()
    for i, move in enumerate(moves):
        board.step(torch.tensor([move]))
    
    legal = board.legal_moves()
    elapsed = time.time() - start
    
    print(f"Legal moves without super-ko: {legal.sum().item()}")
    print(f"Time taken: {elapsed:.3f}s")

if __name__ == "__main__":
    test_basic_super_ko()
    test_without_super_ko()
    print("\nSimple test completed!")
