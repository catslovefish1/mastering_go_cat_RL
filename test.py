import torch
import numpy as np
from engine import GoLegalMoveChecker as legal_module
GoLegalMoveChecker = legal_module.GoLegalMoveChecker

def test_edge_case():
    """Test the edge case progression: board fills up with black, then both players must decide."""
    
    board_size = 5
    checker = GoLegalMoveChecker(board_size=board_size)
    
    print("="*80)
    print("EDGE CASE TEST: Board full of black stones except one position")
    print("="*80)
    
    # Test different empty positions
    empty_positions = [
        (0, 0),  # corner
        (2, 2),  # center
        (board_size-1, board_size-1),  # opposite corner
    ]
    
    for empty_pos in empty_positions:
        print(f"\n{'='*60}")
        print(f"Testing with empty position at {empty_pos}")
        print(f"{'='*60}")
        
        # Create board with all black except one empty
        board = torch.zeros((1, board_size, board_size), dtype=torch.int8)
        board[0, empty_pos[0], empty_pos[1]] = -1  # One empty position
        
        print("\n--- SCENARIO 1: Black's turn (should be suicide, no legal moves) ---")
        current_player = torch.tensor([0], dtype=torch.uint8)  # Black = 0
        
        # Print the board
        print("\nBoard state (0=black, -1=empty, 1=white):")
        board_np = board[0].numpy()
        for row in board_np:
            print(' '.join(f"{cell:2d}" for cell in row))
        
        # Compute legal moves for BLACK
        legal_mask_black, capture_info_black = checker.compute_legal_moves_with_captures(
            board, 
            current_player,
            return_capture_info=True
        )
        
        # Check if black can play at the empty position
        is_legal_black = legal_mask_black[0, empty_pos[0], empty_pos[1]].item()
        has_any_moves_black = legal_mask_black[0].any().item()
        
        print(f"\nBlack's turn:")
        print(f"  Can black play at {empty_pos}? {is_legal_black} (should be False - suicide)")
        print(f"  Does black have ANY legal moves? {has_any_moves_black}")
        print(f"  Legal moves count: {legal_mask_black[0].sum().item()}")
        if has_any_moves_black:
            print(f"  Legal positions: {legal_mask_black[0].nonzero().tolist()}")
        
        print("\n--- SCENARIO 2: White's turn (should be able to capture all) ---")
        current_player = torch.tensor([1], dtype=torch.uint8)  # White = 1
        
        # Compute legal moves for WHITE
        legal_mask_white, capture_info_white = checker.compute_legal_moves_with_captures(
            board, 
            current_player,
            return_capture_info=True
        )
        
        # Check if white can play at the empty position
        is_legal_white = legal_mask_white[0, empty_pos[0], empty_pos[1]].item()
        has_any_moves_white = legal_mask_white[0].any().item()
        
        print(f"\nWhite's turn:")
        print(f"  Can white play at {empty_pos}? {is_legal_white} (should be True - captures)")
        print(f"  Does white have ANY legal moves? {has_any_moves_white}")
        print(f"  Legal moves count: {legal_mask_white[0].sum().item()}")
        if has_any_moves_white:
            print(f"  Legal positions: {legal_mask_white[0].nonzero().tolist()}")
        
        if is_legal_white and capture_info_white['would_capture'][0, empty_pos[0], empty_pos[1]]:
            total_caps = capture_info_white['total_captures'][0, empty_pos[0], empty_pos[1]].item()
            print(f"  Would capture {total_caps} stones (should be {board_size*board_size - 1})")
        
        print("\n--- SCENARIO 3: What if black fills the last spot? ---")
        # Now fill the empty position with black
        board_filled = board.clone()
        board_filled[0, empty_pos[0], empty_pos[1]] = 0  # Black fills last spot
        
        print("\nBoard after black fills last empty (all black now):")
        board_np = board_filled[0].numpy()
        for row in board_np:
            print(' '.join(f"{cell:2d}" for cell in row))
        
        # Check white's options on completely filled board
        current_player = torch.tensor([1], dtype=torch.uint8)  # White = 1
        legal_mask_filled, capture_info_filled = checker.compute_legal_moves_with_captures(
            board_filled, 
            current_player,
            return_capture_info=True
        )
        
        white_legal_count = legal_mask_filled[0].sum().item()
        print(f"\nWhite's legal moves on fully black board: {white_legal_count}")
        print(f"Legal mask:\n{legal_mask_filled[0].numpy().astype(int)}")
        
        if white_legal_count == 0:
            print("  ❌ PROBLEM: White has NO legal moves but should be able to capture dead black group!")
            print("  This is the bug: dead groups (0 liberties) can't be captured by playing on them.")
        else:
            print(f"  ✓ White can play at: {legal_mask_filled[0].nonzero().tolist()}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("-" * 80)
    print("The issue is clear: when the entire board is filled with one color,")
    print("that group has 0 liberties and is technically dead, but the current")
    print("legal move checker only allows moves on EMPTY positions.")
    print("\nWhite should be able to 'capture' by playing on any black stone")
    print("position since the entire black group is dead (0 liberties).")
    print("="*80)

def test_random_agent_behavior():
    """Test what the random agent does in edge cases."""
    print("\n" + "="*80)
    print("RANDOM AGENT BEHAVIOR TEST")
    print("="*80)
    
    from engine.tensor_native_stable import TensorBoard
    from agents.basic import TensorBatchBot
    
    board_size = 5
    device = torch.device('cpu')
    
    # Create a single game
    boards = TensorBoard(
        batch_size=1,
        board_size=board_size,
        history_factor=10,
        device=device,
        enable_timing=False
    )
    
    # Manually set up board: all black except one corner
    boards.board[0] = 0  # All black
    boards.board[0, 0, 0] = -1  # One empty corner
    boards.current_player[0] = 0  # Black's turn
    
    bot = TensorBatchBot(device)
    
    print("\nInitial board (0=black, -1=empty):")
    board_np = boards.board[0].numpy()
    for row in board_np:
        print(' '.join(f"{cell:2d}" for cell in row))
    
    # Test black's move (should pass due to suicide)
    print(f"\nBlack's turn (player {boards.current_player[0].item()}):")
    legal_moves_black = boards.legal_moves()
    print(f"  Legal moves available: {legal_moves_black[0].sum().item()}")
    
    move_black = bot.select_moves(boards)
    print(f"  Agent selects: {move_black[0].tolist()} ({'pass' if move_black[0, 0] < 0 else 'play'})")
    
    # Execute black's move
    boards.step(move_black)
    
    # Test white's move (should capture)
    print(f"\nWhite's turn (player {boards.current_player[0].item()}):")
    legal_moves_white = boards.legal_moves()
    print(f"  Legal moves available: {legal_moves_white[0].sum().item()}")
    if legal_moves_white[0].sum() > 0:
        print(f"  Legal position(s): {legal_moves_white[0].nonzero().tolist()}")
    
    move_white = bot.select_moves(boards)
    print(f"  Agent selects: {move_white[0].tolist()} ({'pass' if move_white[0, 0] < 0 else 'play'})")
    
    # Execute white's move
    boards.step(move_white)
    
    print("\nBoard after white's move:")
    board_np = boards.board[0].numpy()
    for row in board_np:
        print(' '.join(f"{cell:2d}" for cell in row))
    
    print(f"\nPass count: {boards.pass_count[0].item()}")
    print(f"Game over: {boards.is_game_over()[0].item()}")

if __name__ == "__main__":
    test_edge_case()
    print("\n" * 2)
    test_random_agent_behavior()