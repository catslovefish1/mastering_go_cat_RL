"""tensor_batch_duel.py - Tensor-native batch self-play driver for Go.

This module demonstrates efficient batch game simulation patterns:
- Vectorized game management across multiple parallel games
- Efficient progress tracking without breaking GPU pipeline
- Statistical analysis using tensor operations
"""

from __future__ import annotations
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor

# Assuming these follow our refactored interfaces
from engine.tensor_native import TensorBoard
from agents.basic import TensorBatchBot
from interface.ascii import show


# ========================= CONFIGURATION =========================

@dataclass
class SimulationConfig:
    """Configuration for batch game simulation."""
    num_games: int = 100
    board_size: int = 19
    max_moves_factor: float = 10.0  # max_moves = board_size² × factor
    show_boards: int = 0  # Number of final boards to display (0 = none)
    log_interval: int = 100  # Moves between progress updates (0 = silent)
    device: Optional[torch.device] = None
    
    @property
    def max_moves(self) -> int:
        """Calculate maximum moves based on board size."""
        return int(self.board_size * self.board_size * self.max_moves_factor)


# ========================= STATISTICS TRACKING =========================

@dataclass
class GameStatistics:
    """Statistics from batch game simulation."""
    total_games: int
    total_moves: int
    duration_seconds: float
    black_wins: int
    white_wins: int
    draws: int
    final_scores: Tensor  # Shape: (num_games, 2)
    
    @property
    def black_win_rate(self) -> float:
        """Calculate black win percentage."""
        return self.black_wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def white_win_rate(self) -> float:
        """Calculate white win percentage."""
        return self.white_wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def draw_rate(self) -> float:
        """Calculate draw percentage."""
        return self.draws / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def seconds_per_move(self) -> float:
        """Calculate average time per move."""
        return self.duration_seconds / max(1, self.total_moves)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return {
            'total_games': self.total_games,
            'total_moves': self.total_moves,
            'duration_seconds': self.duration_seconds,
            'black_wins': self.black_wins,
            'white_wins': self.white_wins,
            'draws': self.draws,
            'black_win_rate': self.black_win_rate,
            'white_win_rate': self.white_win_rate,
            'draw_rate': self.draw_rate,
            'seconds_per_move': self.seconds_per_move,
        }


# ========================= BATCH SIMULATOR =========================

class BatchGameSimulator:
    """Efficient batch self-play simulator for Go.
    
    This class demonstrates tensor-native patterns for:
    - Managing multiple parallel games
    - Efficient progress tracking
    - Vectorized statistics computation
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize simulator with configuration.
        
        Args:
            config: Simulation configuration parameters
        """
        self.config = config
        self.device = config.device or self._select_device()
    
    @staticmethod
    def _select_device() -> torch.device:
        """Select best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def simulate(self) -> GameStatistics:
        """Run batch simulation and return statistics.
        
        Returns:
            GameStatistics object with simulation results
        """
        print(f"Simulating {self.config.num_games} games on "
              f"{self.config.board_size}×{self.config.board_size} boards...")
        print(f"Device: {self.device}")
        
        # Initialize games and agent
        start_time = time.time()
        boards = self._create_boards()
        bot = self._create_bot()
        
        # Run simulation
        total_moves = self._run_game_loop(boards, bot)
        
        # Compute statistics
        duration = time.time() - start_time
        statistics = self._compute_statistics(boards, total_moves, duration)
        
        # Display results
        self._display_results(statistics)
        self._display_sample_boards(boards)
        
        return statistics
    
    def _create_boards(self) -> TensorBoard:
        """Create batch of game boards."""
        return TensorBoard(
            batch_size=self.config.num_games,
            board_size=self.config.board_size,
            device=self.device
        )
    
    def _create_bot(self) -> TensorBatchBot:
        """Create game-playing agent."""
        return TensorBatchBot(device=self.device)
    
    def _run_game_loop(self, boards: TensorBoard, bot: TensorBatchBot) -> int:
        """Execute main game loop.
        
        Args:
            boards: Batch of game boards
            bot: Agent for move selection
            
        Returns:
            Total number of moves played
        """
        move_count = 0
        
        while move_count < self.config.max_moves:
            # Check if all games are finished
            if boards.is_game_over().all():
                break
            
            # Select and execute moves
            moves = bot.select_moves(boards)
            boards.step(moves)
            move_count += 1
            
            # Log progress
            self._log_progress(boards, move_count)
        
        return move_count
    
    def _log_progress(self, boards: TensorBoard, move_count: int) -> None:
        """Log simulation progress at intervals.
        
        Args:
            boards: Current game states
            move_count: Current move number
        """
        if (self.config.log_interval > 0 and 
            move_count % self.config.log_interval == 0):
            
            # Compute finished games (single GPU sync)
            finished_count = boards.is_game_over().sum().item()
            
            print(f"Move {move_count:4d}: "
                  f"{finished_count}/{self.config.num_games} finished")
    
    def _compute_statistics(
        self, 
        boards: TensorBoard, 
        total_moves: int,
        duration: float
    ) -> GameStatistics:
        """Compute game statistics using tensor operations.
        
        Args:
            boards: Final game states
            total_moves: Total moves played
            duration: Simulation duration in seconds
            
        Returns:
            Computed game statistics
        """
        # Get final scores (single GPU→CPU transfer)
        final_scores = boards.compute_scores().cpu()
        
        # Compute wins using vectorized operations
        black_scores = final_scores[:, 0]
        white_scores = final_scores[:, 1]
        
        black_wins = (black_scores > white_scores).sum().item()
        white_wins = (white_scores > black_scores).sum().item()
        draws = self.config.num_games - black_wins - white_wins
        
        return GameStatistics(
            total_games=self.config.num_games,
            total_moves=total_moves,
            duration_seconds=duration,
            black_wins=black_wins,
            white_wins=white_wins,
            draws=draws,
            final_scores=final_scores
        )
    
    def _display_results(self, stats: GameStatistics) -> None:
        """Display simulation results."""
        print(f"\nFinished in {stats.duration_seconds:.2f}s "
              f"(avg {stats.seconds_per_move:.3f}s per ply)\n")
        
        print(f"Black wins : {stats.black_wins:4d} ({stats.black_win_rate:6.1%})")
        print(f"White wins : {stats.white_wins:4d} ({stats.white_win_rate:6.1%})")
        print(f"Draws      : {stats.draws:4d} ({stats.draw_rate:6.1%})")
        
        # Additional statistics
        avg_black = stats.final_scores[:, 0].mean().item()
        avg_white = stats.final_scores[:, 1].mean().item()
        print(f"\nAverage territory:")
        print(f"  Black: {avg_black:.1f}")
        print(f"  White: {avg_white:.1f}")
    
    def _display_sample_boards(self, boards: TensorBoard) -> None:
        """Display sample final board positions."""
        if self.config.show_boards > 0:
            num_to_show = min(self.config.show_boards, self.config.num_games)
            print(f"\n{'='*50}")
            
            for i in range(num_to_show):
                show(boards, header=f"Game {i + 1} – Final Position", idx=i)


# ========================= CONVENIENCE FUNCTIONS =========================

def simulate_batch_games(
    num_games: int = 100,
    board_size: int = 19,
    **kwargs
) -> GameStatistics:
    """Convenience function for batch game simulation.
    
    Args:
        num_games: Number of parallel games
        board_size: Size of Go board
        **kwargs: Additional configuration options
        
    Returns:
        Game statistics from simulation
    """
    config = SimulationConfig(
        num_games=num_games,
        board_size=board_size,
        **kwargs
    )
    simulator = BatchGameSimulator(config)
    return simulator.simulate()


def run_tournament(
    rounds: int = 10,
    games_per_round: int = 100,
    board_size: int = 19
) -> Dict[str, Any]:
    """Run multiple rounds of games for statistical significance.
    
    Args:
        rounds: Number of simulation rounds
        games_per_round: Games per round
        board_size: Size of Go board
        
    Returns:
        Aggregated statistics across all rounds
    """
    print(f"Running tournament: {rounds} rounds × {games_per_round} games")
    
    all_stats = []
    total_black_wins = 0
    total_white_wins = 0
    total_draws = 0
    
    for round_num in range(rounds):
        print(f"\n{'='*50}")
        print(f"Round {round_num + 1}/{rounds}")
        print(f"{'='*50}")
        
        stats = simulate_batch_games(
            num_games=games_per_round,
            board_size=board_size,
            log_interval=0,  # Silent for tournament
            show_boards=0
        )
        
        all_stats.append(stats)
        total_black_wins += stats.black_wins
        total_white_wins += stats.white_wins
        total_draws += stats.draws
    
    # Aggregate results
    total_games = rounds * games_per_round
    
    return {
        'rounds': rounds,
        'games_per_round': games_per_round,
        'total_games': total_games,
        'total_black_wins': total_black_wins,
        'total_white_wins': total_white_wins,
        'total_draws': total_draws,
        'black_win_rate': total_black_wins / total_games,
        'white_win_rate': total_white_wins / total_games,
        'draw_rate': total_draws / total_games,
        'individual_rounds': [s.to_dict() for s in all_stats]
    }


# ========================= MAIN EXECUTION =========================

def main():
    """Main entry point with example usage."""
    # Example 1: Simple batch simulation
    print("Example 1: Quick test on 9×9 boards")
    simulate_batch_games(
        num_games=64,
        board_size=9,
        show_boards=3,
        log_interval=50
    )
    
    # Example 2: Larger simulation
    print("\n\nExample 2: Standard 19×19 simulation")
    stats = simulate_batch_games(
        num_games=256,
        board_size=19,
        show_boards=0,
        log_interval=100
    )
    
    # Example 3: Tournament mode (commented out for speed)
    # print("\n\nExample 3: Tournament mode")
    # tournament_results = run_tournament(
    #     rounds=5,
    #     games_per_round=100,
    #     board_size=13
    # )
    # print(f"\nTournament Summary:")
    # print(f"Black win rate: {tournament_results['black_win_rate']:.1%}")
    # print(f"White win rate: {tournament_results['white_win_rate']:.1%}")


if __name__ == "__main__":
    main()