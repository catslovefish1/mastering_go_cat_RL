"""
Random agent vs MCTS agent self-play driver.

This script runs games between:
- Black: Random agent (from agents.basic.TensorBatchBot)
- White: MCTS agent (from agents.mcts)

Run:
    python random_vs_mcts.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
from torch import Tensor

# Import TensorBoard from engine
from engine.tensor_native import TensorBoard
BatchTensor = Tensor

# Import agents
from agents.basic import TensorBatchBot  # Random agent
from agents.mcts import create_pure_mcts_agent, create_neural_mcts_agent  # MCTS agents

# Import visualization
from interface.ascii import show

# Import shared utilities
from utils.shared import select_device, get_batch_indices

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Parameters controlling the random vs MCTS simulation."""

    num_games: int = 100                # games in parallel
    board_size: int = 9                 # Go board edge length (9x9 for faster games)
    max_moves_factor: int = 100         # ply limit = board² × factor
    show_boards: int = 0                # number of final boards to print
    log_interval: int = 50              # progress print frequency (ply)
    
    # Agent configuration
    mcts_simulations: int = 50          # MCTS simulations per move
    mcts_use_network: bool = False      # Whether MCTS uses neural network
    mcts_plays_white: bool = True       # If True, MCTS plays white; else black
    
    device: Optional[torch.device | str] = None  # override auto-device

    @property
    def max_moves(self) -> int:
        """Maximum game length in plies before forced termination."""
        return self.board_size ** 2 * self.max_moves_factor

# -----------------------------------------------------------------------------
# Statistics container
# -----------------------------------------------------------------------------
@dataclass
class GameStatistics:
    """Aggregated results of a simulation run."""

    total_games: int
    total_moves: int
    duration_seconds: float
    black_wins: int  # Random agent wins (if mcts_plays_white=True)
    white_wins: int  # MCTS agent wins (if mcts_plays_white=True)
    draws: int
    final_scores: Tensor  # shape (B, 2)
    
    # Agent info
    black_agent: str
    white_agent: str

    # win/draw rates -----------------------------------------------------------
    @property
    def black_win_rate(self) -> float: 
        return self.black_wins / self.total_games

    @property
    def white_win_rate(self) -> float: 
        return self.white_wins / self.total_games

    @property
    def draw_rate(self) -> float: 
        return self.draws / self.total_games

    @property
    def seconds_per_move(self) -> float:
        return self.duration_seconds / max(1, self.total_moves)

    # helper for structured logging -------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for logging."""
        return {
            **self.__dict__,
            "black_win_rate": self.black_win_rate,
            "white_win_rate": self.white_win_rate,
            "draw_rate": self.draw_rate,
            "seconds_per_move": self.seconds_per_move,
        }

# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------
class RandomVsMCTSSimulator:
    """Drives games between random and MCTS agents."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        # Use shared device selection utility
        self.device = torch.device(cfg.device) if cfg.device else select_device()

    # ------------------------- public API ------------------------------------
    def simulate(self) -> GameStatistics:
        """Run batch simulation and return statistics.
        
        Returns:
            GameStatistics object with results
        """
        self._print_header()
        
        # Initialize boards
        boards = TensorBoard(self.cfg.num_games, self.cfg.board_size, self.device)
        
        # Initialize agents
        random_bot, mcts_bot, black_agent_name, white_agent_name = self._create_agents()

        # Run simulation
        t0 = time.time()
        with torch.no_grad():  # inference mode – gradients disabled
            moves_made = self._play_games(boards, random_bot, mcts_bot)
        elapsed = time.time() - t0

        # Collect and display results
        stats = self._collect_stats(boards, moves_made, elapsed, black_agent_name, white_agent_name)
        self._print_summary(stats)
        self._maybe_show_boards(boards)
        
        return stats

    # ------------------------- agent creation --------------------------------
    def _create_agents(self) -> Tuple[Any, Any, str, str]:
        """Create the random and MCTS agents.
        
        Returns:
            Tuple of (random_bot, mcts_bot, black_agent_name, white_agent_name)
        """
        # Create random agent
        random_bot = TensorBatchBot(self.device)
        
        # Create MCTS agent
        if self.cfg.mcts_use_network:
            mcts_bot = create_neural_mcts_agent(
                simulations=self.cfg.mcts_simulations,
                device=self.device
            )
            mcts_name = f"Neural MCTS ({self.cfg.mcts_simulations} sims)"
        else:
            mcts_bot = create_pure_mcts_agent(
                simulations=self.cfg.mcts_simulations,
                device=self.device
            )
            mcts_name = f"Pure MCTS ({self.cfg.mcts_simulations} sims)"
        
        # Determine which agent plays which color
        if self.cfg.mcts_plays_white:
            black_agent_name = "Random"
            white_agent_name = mcts_name
        else:
            black_agent_name = mcts_name
            white_agent_name = "Random"
        
        return random_bot, mcts_bot, black_agent_name, white_agent_name

    # ------------------------- core loop -------------------------------------
    def _play_games(self, boards: TensorBoard, random_bot: Any, mcts_bot: Any) -> int:
        """Play games until completion or move limit.
        
        Args:
            boards: TensorBoard instance
            random_bot: Random agent
            mcts_bot: MCTS agent
            
        Returns:
            Total number of plies played
        """
        finished = boards.is_game_over()  # (B,) bool on device
        ply = 0
        
        while ply < self.cfg.max_moves and not finished.all():
            # Determine which agent plays based on current player
            current_players = boards.current_player
            
            # Select agent based on configuration and current player
            if self.cfg.mcts_plays_white:
                # Random plays black (0), MCTS plays white (1)
                use_mcts = (current_players == 1)
            else:
                # MCTS plays black (0), Random plays white (1)
                use_mcts = (current_players == 0)
            
            # Get moves from both agents
            if use_mcts.any() and (~use_mcts).any():
                # Both agents need to play
                moves = torch.zeros((boards.batch_size, 2), dtype=torch.int32, device=self.device)
                
                # Get moves from MCTS for its games
                mcts_moves = mcts_bot.select_moves(boards)
                moves[use_mcts] = mcts_moves[use_mcts]
                
                # Get moves from random for its games
                random_moves = random_bot.select_moves(boards)
                moves[~use_mcts] = random_moves[~use_mcts]
            elif use_mcts.all():
                # Only MCTS plays
                moves = mcts_bot.select_moves(boards)
            else:
                # Only random plays
                moves = random_bot.select_moves(boards)
            
            # Make moves
            boards.step(moves)
            
            # Update finished status (in-place OR to avoid extra allocation)
            finished |= boards.is_game_over()
            ply += 1

            # Progress logging
            self._maybe_log_progress(ply, finished)
        
        return ply

    # ------------------------- analysis --------------------------------------
    def _collect_stats(
        self, 
        boards: TensorBoard, 
        moves: int, 
        dt: float,
        black_agent_name: str,
        white_agent_name: str
    ) -> GameStatistics:
        """Collect game statistics from completed games.
        
        Args:
            boards: Completed game boards
            moves: Total moves made
            dt: Duration in seconds
            black_agent_name: Name of black player
            white_agent_name: Name of white player
            
        Returns:
            GameStatistics object
        """
        # Get final scores
        scores = boards.compute_scores().cpu()
        black_scores = scores[:, 0]
        white_scores = scores[:, 1]
        
        # Count outcomes
        black_wins = (black_scores > white_scores).sum().item()
        white_wins = (white_scores > black_scores).sum().item()
        draws = self.cfg.num_games - black_wins - white_wins
        
        return GameStatistics(
            total_games=self.cfg.num_games,
            total_moves=moves,
            duration_seconds=dt,
            black_wins=black_wins,
            white_wins=white_wins,
            draws=draws,
            final_scores=scores,
            black_agent=black_agent_name,
            white_agent=white_agent_name
        )

    # ------------------------- output ----------------------------------------
    def _print_header(self) -> None:
        """Print simulation header."""
        if self.cfg.mcts_plays_white:
            matchup = "Random (Black) vs MCTS (White)"
        else:
            matchup = "MCTS (Black) vs Random (White)"
        
        print(
            f"Running {self.cfg.num_games} games: {matchup}\n"
            f"Board: {self.cfg.board_size}×{self.cfg.board_size} | "
            f"Device: {self.device} | "
            f"MCTS simulations: {self.cfg.mcts_simulations}"
        )

    def _maybe_log_progress(self, ply: int, finished: BatchTensor) -> None:
        """Log progress if interval reached.
        
        Args:
            ply: Current ply number
            finished: Boolean tensor of finished games
        """
        if self.cfg.log_interval and ply % self.cfg.log_interval == 0:
            # Single sync for efficiency
            finished_count = finished.sum().item()
            print(
                f"Ply {ply:4d}: {finished_count}/"
                f"{self.cfg.num_games} finished"
            )

    @staticmethod
    def _print_summary(stats: GameStatistics) -> None:
        """Print game statistics summary.
        
        Args:
            stats: GameStatistics object
        """
        print(
            f"\nFinished {stats.total_games} games in {stats.duration_seconds:.2f}s "
            f"({stats.seconds_per_move:.4f}s/ply)"
        )
        print(f"\n{stats.black_agent} (Black) wins: {stats.black_wins:4d} ({stats.black_win_rate:6.1%})")
        print(f"{stats.white_agent} (White) wins: {stats.white_wins:4d} ({stats.white_win_rate:6.1%})")
        print(f"Draws: {stats.draws:4d} ({stats.draw_rate:6.1%})\n")
        
        # Determine winner
        if stats.black_wins > stats.white_wins:
            print(f"🏆 {stats.black_agent} (Black) wins the match!")
        elif stats.white_wins > stats.black_wins:
            print(f"🏆 {stats.white_agent} (White) wins the match!")
        else:
            print("🤝 The match is a draw!")

    def _maybe_show_boards(self, boards: TensorBoard) -> None:
        """Display final boards if requested.
        
        Args:
            boards: Final game boards
        """
        for i in range(min(self.cfg.show_boards, self.cfg.num_games)):
            show(boards, header=f"Game {i + 1}", idx=i)

# -----------------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------------

def random_vs_mcts(
    num_games: int = 100,
    board_size: int = 9,
    mcts_simulations: int = 50,
    mcts_use_network: bool = False,
    mcts_plays_white: bool = True,
    **kwargs
) -> GameStatistics:
    """Run games between random and MCTS agents.
    
    Args:
        num_games: Number of parallel games
        board_size: Board dimension
        mcts_simulations: MCTS simulations per move
        mcts_use_network: Whether MCTS uses neural network
        mcts_plays_white: If True, MCTS plays white; else black
        **kwargs: Additional SimulationConfig parameters
        
    Returns:
        GameStatistics object with results
    """
    config = SimulationConfig(
        num_games=num_games,
        board_size=board_size,
        mcts_simulations=mcts_simulations,
        mcts_use_network=mcts_use_network,
        mcts_plays_white=mcts_plays_white,
        **kwargs
    )
    simulator = RandomVsMCTSSimulator(config)
    return simulator.simulate()

# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Entry point for command-line execution."""
    # Test 1: Random vs Pure MCTS (50 simulations)
    print("=" * 60)
    print("Test 1: Random vs Pure MCTS")
    print("=" * 60)
    random_vs_mcts(
        num_games=3,
        board_size=3,
        mcts_simulations=2,
        mcts_use_network=False,
        mcts_plays_white=True,
        show_boards=2,
        log_interval=1,
    )
    
    # # Test 2: Random vs Neural MCTS (100 simulations)
    # print("\n" + "=" * 60)
    # print("Test 2: Random vs Neural MCTS")
    # print("=" * 60)
    # random_vs_mcts(
    #     num_games=100,
    #     board_size=9,
    #     mcts_simulations=100,
    #     mcts_use_network=True,
    #     mcts_plays_white=True,
    #     show_boards=2,
    #     log_interval=50
    # )
    
    # # Test 3: Switch colors - MCTS plays black
    # print("\n" + "=" * 60)
    # print("Test 3: MCTS (Black) vs Random (White)")
    # print("=" * 60)
    # random_vs_mcts(
    #     num_games=100,
    #     board_size=9,
    #     mcts_simulations=50,
    #     mcts_use_network=False,
    #     mcts_plays_white=False,
    #     show_boards=2,
    #     log_interval=50
    # )

if __name__ == "__main__":
    main()