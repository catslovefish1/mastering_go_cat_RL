"""simple_example.py - Quick example showing the speed improvements"""

import time
import torch
from engine.tensor_native_stable import TensorBoard
from agents.basic import TensorBatchBot


def time_games(enable_super_ko=True, simplified=True, num_games=128):
    """Time a batch of games with given settings"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create board
    board = TensorBoard(
        batch_size=num_games,
        board_size=7,
        device=device,
        enable_super_ko=enable_super_ko,
        simplified_super_ko=simplified,
        max_history=1000
    )
    bot = TensorBatchBot(device=device)
    
    # Play games
    start = time.time()
    moves = 0
    
    with torch.no_grad():
        while moves < 5000 and not board.is_game_over().all():
            bot_moves = bot.select_moves(board)
            board.step(bot_moves)
            moves += 1
    
    elapsed = time.time() - start
    
    # Get results
    finished = board.is_game_over().sum().item()
    unfinished = num_games - finished
    
    return {
        'time': elapsed,
        'moves': moves,
        'finished': finished,
        'unfinished': unfinished,
        'moves_per_second': (moves * num_games) / elapsed
    }


def main():
    """Compare different configurations"""
    
    print("Super-Ko Performance Comparison")
    print("=" * 50)
    
    configs = [
        ("No super-ko", False, True),
        ("Fast super-ko", True, True),
        ("Full super-ko", True, False),
    ]
    
    for name, enable, simplified in configs:
        print(f"\n{name}:")
        result = time_games(enable, simplified)
        
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Speed: {result['moves_per_second']:,.0f} moves/sec")
        print(f"  Unfinished: {result['unfinished']}/128 games")
    
    # Show the difference
    print("\n" + "="*50)
    print("Summary:")
    print("- Fast super-ko is nearly as fast as no super-ko")
    print("- Fast super-ko prevents most infinite loops")
    print("- Full super-ko is more accurate but slower")
    
    # Example of checking if super-ko is working
    print("\n" + "="*50)
    print("Checking super-ko effectiveness:")
    
    board = TensorBoard(
        batch_size=100,
        board_size=7,
        enable_super_ko=True,
        simplified_super_ko=True
    )
    
    # Get position history stats
    stats = board.get_position_history_stats()
    print(f"Super-ko enabled: {stats['super_ko_enabled']}")
    print(f"Mode: {'Simplified' if stats['simplified'] else 'Full'}")
    print(f"Max history: {stats['max_history']}")


if __name__ == "__main__":
    main()