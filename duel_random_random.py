"""
Batch self-play driver leveraging TensorBatchBot with timing analysis.

Run:
    python duel.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import Tensor

from engine.tensor_native import TensorBoard, select_device
from agents.basic import TensorBatchBot
from interface.ascii import show

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Parameters controlling the batch self-play simulation."""

    num_games: int = 100                # games in parallel
    board_size: int = 19                # Go board edge length
    max_moves_factor: int = 10         # ply limit = board² × factor
    show_boards: int = 0                # number of final boards to print
    log_interval: int = 100             # progress print frequency (ply)
    device: Optional[torch.device | str] = None  # override auto-device
    enable_timing: bool = True          # enable timing analysis

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
    black_wins: int
    white_wins: int
    draws: int
    final_scores: Tensor  # shape (B, 2)

    # win/draw rates -----------------------------------------------------------
    @property
    def black_win_rate(self) -> float: return self.black_wins / self.total_games

    @property
    def white_win_rate(self) -> float: return self.white_wins / self.total_games

    @property
    def draw_rate(self) -> float: return self.draws / self.total_games

    @property
    def seconds_per_move(self) -> float:
        return self.duration_seconds / max(1, self.total_moves)

    # helper for structured logging -------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
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
class BatchGameSimulator:
    """Drives a batch of self-play Go games on a chosen device."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device) if cfg.device else select_device()

    # ------------------------- public API ------------------------------------
    def simulate(self) -> GameStatistics:
        print(
            f"Running {self.cfg.num_games} games on "
            f"{self.cfg.board_size}×{self.cfg.board_size} ({self.device})"
        )

        # Create board with timing enabled
        boards = TensorBoard(
            self.cfg.num_games, 
            self.cfg.board_size, 
            self.device,
            enable_timing=self.cfg.enable_timing
        )
        bot = TensorBatchBot(self.device)
        
        boards.print_union_find_grid(batch_idx=0, column=0)
        boards.print_union_find_grid(batch_idx=0, column=1)
        boards.print_union_find_grid(batch_idx=0, column=2)
        
        t0 = time.time()
        with torch.no_grad():  # inference mode – gradients disabled
            moves_made = self._play_games(boards, bot)
        elapsed = time.time() - t0

        stats = self._collect_stats(boards, moves_made, elapsed)
        self._print_summary(stats)
        self._maybe_show_boards(boards)
        
        # Print timing report if enabled
        if self.cfg.enable_timing and hasattr(boards, 'print_timing_report'):
            boards.print_timing_report(top_n=30)
            
            # Additional performance metrics
            print("\n" + "="*80)
            print("PERFORMANCE METRICS")
            print("="*80)
            print(f"Total simulation time: {elapsed:.2f} seconds")
            print(f"Moves per second: {moves_made/elapsed:.1f}")
            print(f"Games per second: {self.cfg.num_games/elapsed:.1f}")
            print(f"Time per move: {elapsed/moves_made*1000:.2f} ms")
            print(f"Time per game: {elapsed/self.cfg.num_games:.3f} seconds")
            

        
        return stats

    # ------------------------- core loop -------------------------------------
    def _play_games(self, boards: TensorBoard, bot: TensorBatchBot) -> int:
        finished = boards.is_game_over()  # (B,) bool on device
        ply = 0
        
        for ply in range(self.cfg.max_moves):
            boards.step(bot.select_moves(boards))
            finished |= boards.is_game_over()  # inplace OR, no extra alloc
            ply += 1

            if self.cfg.log_interval and ply % self.cfg.log_interval == 0:
                finished_count = finished.sum().cpu()  # Transfer to CPU (non-blocking)
                print(
                  f"Ply {ply:4d}: {finished_count}/"  # No .item() - Python converts tensor to string
                   f"{self.cfg.num_games} finished"
                )
        return ply
    
    
    
    

    # ------------------------- analysis --------------------------------------
    def _collect_stats(
        self, boards: TensorBoard, moves: int, dt: float
    ) -> GameStatistics:
        scores = boards.compute_scores().cpu()
        black = scores[:, 0]
        white = scores[:, 1]
        bw = (black > white).sum().item()
        ww = (white > black).sum().item()
        draws = self.cfg.num_games - bw - ww
        return GameStatistics(
            self.cfg.num_games, moves, dt, bw, ww, draws, scores
        )

    # ------------------------- output ----------------------------------------
    @staticmethod
    def _print_summary(s: GameStatistics) -> None:
        print(
            f"\nFinished {s.total_games} games in {s.duration_seconds:.2f}s "
            f"({s.seconds_per_move:.4f}s/ply)"
        )
        print(f"Black wins: {s.black_wins:4d} ({s.black_win_rate:6.1%})")
        print(f"White wins: {s.white_wins:4d} ({s.white_win_rate:6.1%})")
        print(f"Draws     : {s.draws:4d} ({s.draw_rate:6.1%})\n")

    def _maybe_show_boards(self, boards: TensorBoard) -> None:
        for i in range(min(self.cfg.show_boards, self.cfg.num_games)):
            show(boards, header=f"Game {i + 1}", idx=i)

# -----------------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------------

def simulate_batch_games(num_games=100, board_size=19, enable_timing=True, **kwargs) -> GameStatistics:
    """Run a batch of games with minimal boilerplate."""
    return BatchGameSimulator(
        SimulationConfig(num_games, board_size, enable_timing=enable_timing, **kwargs)
    ).simulate()

# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------

def main() -> None:
    # Run with timing analysis enabled
    print("Running batch Go simulation with timing analysis...")
    print("="*80)
    

    
    # Main timing run
    print("\nStarting main timing analysis run...")
    simulate_batch_games(
        num_games=1,  # 512 games
        board_size=7,     # 9x9 board
        show_boards=2,    # Don't show boards
        log_interval=64,  # Log every 64 moves
        enable_timing=True  # Enable timing
    )

if __name__ == "__main__":
    main()