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
    show_boards=0,
    log_interval=64,
    enable_timing=True,
    save_history=True,  # Save game histories to JSON
    num_games_to_save=5  # Number of games to save
):
    """Run batch Go games."""
    device = select_device()
    print(f"Running {num_games} games on {board_size}×{board_size} ({device})")
    
    # Create boards and bot
    boards = TensorBoard(num_games, board_size, device, enable_timing)
    bot = TensorBatchBot(device)
    
    # Play games
    t0 = time.time()
    with torch.no_grad():
        finished = boards.is_game_over()
        ply = 0
        
        while not finished.all() and ply < board_size * board_size * 4:
            boards.step(bot.select_moves(boards))
            finished |= boards.is_game_over()
            ply += 1
            
            if log_interval and ply % log_interval == 0:
                print(f"Ply {ply:4d}: {finished.sum()}/{num_games} finished")
    
    elapsed = time.time() - t0
    
    # Results
    scores = boards.compute_scores().cpu()
    black_wins = (scores[:, 0] > scores[:, 1]).sum().item()
    white_wins = (scores[:, 1] > scores[:, 0]).sum().item()
    draws = num_games - black_wins - white_wins
    
    print(f"\nFinished in {elapsed:.2f}s ({ply} moves)")
    print(f"Black wins: {black_wins} ({black_wins/num_games:.1%})")
    print(f"White wins: {white_wins} ({white_wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")
    
    # Save game histories to JSON
    if save_history:
        save_game_histories_to_json(boards, num_games_to_save=num_games_to_save)
    
    # Timing analysis
    if enable_timing:
        print_timing_report(boards)
        print_performance_metrics(elapsed, ply, num_games)
    
    # Show final boards
    if show_boards > 0:
        print(f"\nShowing {min(show_boards, num_games)} final board positions:")
        step = max(1, num_games // show_boards)
        for i in range(0, min(show_boards * step, num_games), step):
            show(boards, header=f"Game {i + 1}", idx=i)

if __name__ == "__main__":
    simulate_batch_games(
        num_games=10,
        board_size=11,
        show_boards=2,
        enable_timing=False,
        save_history=True,
        num_games_to_save=5  # Save first 5 games
    )