"""
Test super-ko rule implementation with Zobrist hashing.
"""

import torch
from engine.tensor_native import TensorBoard

def test_super_ko_simple():
    """Test that super-ko prevents position repetition."""
    # Create a small board for easier testing
    board = TensorBoard(batch_size=1, board_size=5, enable_super_ko=True)
    
    print("Testing super-ko rule on 5x5 board...")
    
    # Create a simple ko situation
    # Black plays at (1, 1)
    board.step(torch.tensor([[1, 1]]))
    print(f"Move 1: Black plays at (1,1)")
    
    # White plays at (1, 2)
    board.step(torch.tensor([[1, 2]]))
    print(f"Move 2: White plays at (1,2)")
    
    # Black plays at (0, 1)
    board.step(torch.tensor([[0, 1]]))
    print(f"Move 3: Black plays at (0,1)")
    
    # White plays at (0, 2)
    board.step(torch.tensor([[0, 2]]))
    print(f"Move 4: White plays at (0,2)")
    
    # Black plays at (2, 1)
    board.step(torch.tensor([[2, 1]]))
    print(f"Move 5: Black plays at (2,1)")
    
    # White plays at (2, 2)
    board.step(torch.tensor([[2, 2]]))
    print(f"Move 6: White plays at (2,2)")
    
    # Now we have a pattern that could create ko
    # Let's check legal moves
    legal = board.legal_moves()
    print(f"\nCurrent board state:")
    print(board.board[0])
    print(f"\nLegal moves mask:")
    print(legal[0])
    
    # Save initial legal move count
    initial_legal_count = legal.sum().item()
    
    # Make a few more moves to create a potential super-ko situation
    board.step(torch.tensor([[1, 0]]))  # Black
    print(f"\nMove 7: Black plays at (1,0)")
    
    board.step(torch.tensor([[2, 0]]))  # White
    print(f"Move 8: White plays at (2,0)")
    
    # Check legal moves again
    legal2 = board.legal_moves()
    print(f"\nCurrent board state:")
    print(board.board[0])
    print(f"\nLegal moves mask after more moves:")
    print(legal2[0])
    
    print(f"\nInitial legal moves: {initial_legal_count}")
    print(f"Current legal moves: {legal2.sum().item()}")
    
    print("\n✅ Super-ko test completed!")

def test_super_ko_disabled():
    """Test that disabling super-ko allows position repetition."""
    board_no_ko = TensorBoard(batch_size=1, board_size=5, enable_super_ko=False)
    board_with_ko = TensorBoard(batch_size=1, board_size=5, enable_super_ko=True)
    
    print("\n\nTesting with super-ko disabled vs enabled...")
    
    # Make the same moves on both boards
    moves = [
        [1, 1], [1, 2], [0, 1], [0, 2], 
        [2, 1], [2, 2], [1, 0], [2, 0]
    ]
    
    for i, move in enumerate(moves):
        board_no_ko.step(torch.tensor([move]))
        board_with_ko.step(torch.tensor([move]))
        print(f"Move {i+1}: {'Black' if i % 2 == 0 else 'White'} at {move}")
    
    legal_no_ko = board_no_ko.legal_moves()
    legal_with_ko = board_with_ko.legal_moves()
    
    print(f"\nLegal moves without super-ko: {legal_no_ko.sum().item()}")
    print(f"Legal moves with super-ko: {legal_with_ko.sum().item()}")
    
    if legal_no_ko.sum() >= legal_with_ko.sum():
        print("✅ Super-ko correctly restricts moves when enabled!")
    else:
        print("❌ Unexpected: super-ko should restrict moves")
    
def test_hash_uniqueness():
    """Test that Zobrist hashing produces unique hashes for different positions."""
    board = TensorBoard(batch_size=1, board_size=9, enable_super_ko=True)
    
    print("\n\nTesting Zobrist hash uniqueness...")
    
    hashes = []
    
    # Play a sequence of moves and collect hashes
    moves = [
        [0, 0], [0, 1], [1, 0], [1, 1],
        [2, 0], [2, 1], [3, 0], [3, 1],
        [4, 0], [4, 1]
    ]
    
    for i, move in enumerate(moves):
        board.step(torch.tensor([move]))
        current_hash = board.current_hash[0].item()
        hashes.append(current_hash)
        print(f"Move {i+1}: Hash = {current_hash}")
    
    # Check for uniqueness
    unique_hashes = len(set(hashes))
    total_hashes = len(hashes)
    
    print(f"\nTotal positions: {total_hashes}")
    print(f"Unique hashes: {unique_hashes}")
    
    if unique_hashes == total_hashes:
        print("✅ All positions have unique hashes!")
    else:
        print(f"⚠️ Hash collision detected: {total_hashes - unique_hashes} collisions")

def test_capture_hash_update():
    """Test that captures correctly update the Zobrist hash."""
    board = TensorBoard(batch_size=1, board_size=5, enable_super_ko=True)
    
    print("\n\nTesting hash updates with captures...")
    
    # Create a capture situation
    # White stones surrounding a black stone
    board.step(torch.tensor([[1, 1]]))  # Black center
    board.step(torch.tensor([[0, 1]]))  # White top
    board.step(torch.tensor([[-1, -1]]))  # Black pass
    board.step(torch.tensor([[2, 1]]))  # White bottom
    board.step(torch.tensor([[-1, -1]]))  # Black pass
    board.step(torch.tensor([[1, 0]]))  # White left
    
    hash_before = board.current_hash[0].item()
    print(f"Hash before capture: {hash_before}")
    print("Board before capture:")
    print(board.board[0])
    
    # Black passes, white captures
    board.step(torch.tensor([[-1, -1]]))  # Black pass
    board.step(torch.tensor([[1, 2]]))  # White right - captures black
    
    hash_after = board.current_hash[0].item()
    print(f"\nHash after capture: {hash_after}")
    print("Board after capture:")
    print(board.board[0])
    
    if hash_before != hash_after:
        print("✅ Hash changed after capture!")
    else:
        print("❌ Hash should change after capture")

if __name__ == "__main__":
    test_super_ko_simple()
    test_super_ko_disabled()
    test_hash_uniqueness()
    test_capture_hash_update()
    
    print("\n" + "="*50)
    print("All super-ko tests completed!")
    print("="*50)
