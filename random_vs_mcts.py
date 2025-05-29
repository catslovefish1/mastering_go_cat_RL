"""
Random agent vs MCTS agent driver.

Run:
    python random_vs_mcts.py
"""
from __future__ import annotations
import time
import torch

# Import engine and agents
from engine.tensor_native import TensorBoard
from agents.basic import TensorBatchBot
from agents.simple_mcts import create_simple_mcts_agent
from interface.ascii import show
from utils.shared import select_device


def main():
    """Run Random vs MCTS match."""
    device = select_device()
    

    
    # Game settings
    num_games = 100
    simulations =10
    board_size = 5
    
    # Create boards
    boards = TensorBoard(num_games, board_size, device)

    # Create agents
    random_agent = TensorBatchBot(device)
    mcts_agent = create_simple_mcts_agent(simulations, device=device)
    
    print(f"\n{'='*50}")
    print(f"Random (Black) vs MCTS-{simulations} (White)")
    print(f"{num_games} games on {board_size}x{board_size} board")
    print(f"{'='*50}")
    
    start_time = time.time()
    move_count = 0
    max_moves = board_size * board_size * 2
    
    # Play games
    while move_count < max_moves and not boards.is_game_over().all():
        current_players = boards.current_player
        game_over = boards.is_game_over()
        
        # Black (Random) plays
        black_mask = (current_players == 0) & ~game_over
        # White (MCTS) plays  
        white_mask = (current_players == 1) & ~game_over
        
        # Get moves
        moves = torch.zeros((num_games, 2), dtype=torch.int32, device=device)
        
        if black_mask.any():
            moves[black_mask] = random_agent.select_moves(boards)[black_mask]
        
        if white_mask.any():
            moves[white_mask] = mcts_agent.select_moves(boards)[white_mask]
        
        boards.step(moves)
        move_count += 1
        
        if move_count % 50 == 0:
            print(f"Move {move_count}: {game_over.sum().item()}/{num_games} finished")
    
    # Get results
    duration = time.time() - start_time
    scores = boards.compute_scores()
    
    black_wins = (scores[:, 0] > scores[:, 1]).sum().item()
    white_wins = (scores[:, 1] > scores[:, 0]).sum().item()
    draws = num_games - black_wins - white_wins
    
    # Print results
    print(f"\nCompleted in {duration:.2f}s")
    print(f"\nRandom (Black): {black_wins} wins ({100*black_wins/num_games:.1f}%)")
    print(f"MCTS-50 (White): {white_wins} wins ({100*white_wins/num_games:.1f}%)")
    print(f"Draws: {draws} ({100*draws/num_games:.1f}%)")
    
    if black_wins > white_wins:
        print(f"\n🏆 Random wins!")
    elif white_wins > black_wins:
        print(f"\n🏆 MCTS wins!")
    else:
        print("\n🤝 It's a draw!")
    
    # Show 2 final boards
    for i in range(min(2, num_games)):
        show(boards, header=f"Game {i+1}", idx=i)


if __name__ == "__main__":
    main()