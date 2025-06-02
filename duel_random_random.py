"""Minimal Go simulation driver with JSON history saving."""
import time
import torch
from engine.tensor_native import TensorBoard
from agents.basic import TensorBatchBot
from interface.ascii import show
from utils.shared import (
    select_device, 
    print_timing_report, 
    print_performance_metrics,
    save_game_histories_to_json
)

def simulate_batch_games(
    num_games=512,
    board_size=9,
    history_factor=10,  # NEW: Controls history depth
    show_boards=0,
    log_interval=10,
    enable_timing=True,
    save_history=True,
    num_games_to_save=5
):
    """Run batch Go games."""
    device = select_device()
    print(f"Running {num_games} games on {board_size}×{board_size} ({device})")
    
    # Create boards and bot
    boards = TensorBoard(num_games, board_size, history_factor, device, enable_timing)
    bot = TensorBatchBot(device)
    
    # Play games
    t0 = time.time()
    with torch.no_grad():
        ply = 0
        print( f"Ply {ply:4d}: start" )
        
        while ply < board_size * board_size * history_factor:
            finished = boards.is_game_over()
                
            moves = bot.select_moves(boards)
            boards.step(moves)
            ply += 1
            
            # Log progress
            if log_interval and ply % log_interval == 0:
                finished = boards.is_game_over()
                finished_count = finished.sum().item()
                print(f"Ply {ply:4d}: {finished_count}/{num_games} finished")
     
    elapsed = time.time() - t0
    

   # Results (keep on GPU until final .item())
    scores = boards.compute_scores()
    black_wins = (scores[:, 0] > scores[:, 1]).sum().cpu()
    white_wins = (scores[:, 1] > scores[:, 0]).sum().cpu()
    draws = num_games - black_wins - white_wins
    
    print(f"\nFinished in {elapsed:.2f}s ({ply} moves)")
    print(f"Black wins: {black_wins} ({black_wins/num_games:.1%})")
    print(f"White wins: {white_wins} ({white_wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")
    
    # Save game histories
    if save_history:
        save_game_histories_to_json(boards, num_games_to_save=num_games_to_save)
    
    # Timing
    if enable_timing:
        print_timing_report(boards)
        print_performance_metrics(elapsed, ply, num_games)
    
    # Show boards
    if show_boards > 0:
        print(f"\nShowing {min(show_boards, num_games)} final boards:")
        step = max(1, num_games // show_boards)
        for i in range(0, min(show_boards * step, num_games), step):
            show(boards, header=f"Game {i + 1}", idx=i)

if __name__ == "__main__":
    simulate_batch_games(
        num_games=2**10,
        board_size=19,
        history_factor=4,
        log_interval=2**4,
        show_boards=2,
        enable_timing=True,
        save_history=True,
        num_games_to_save=5
    )