"""
tensor_batch_duel.py — batch self-play driver (optimised, no profiler)
----------------------------------------------------------------------
Usage:
    python tensor_batch_duel.py
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import Tensor

from engine.tensor_native import TensorBoard
from agents.basic import TensorBatchBot
from interface.ascii import show

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class SimulationConfig:
    num_games: int = 100
    board_size: int = 19
    max_moves_factor: int= 100        # max plies = board² × factor
    show_boards: int = 0
    log_interval: int = 100
    device: Optional[torch.device] = None

    @property
    def max_moves(self) -> int:
        return int(self.board_size * self.board_size * self.max_moves_factor)

# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
@dataclass
class GameStatistics:
    total_games: int
    total_moves: int
    duration_seconds: float
    black_wins: int
    white_wins: int
    draws: int
    final_scores: Tensor        # (B, 2)

    @property
    def black_win_rate(self) -> float: return self.black_wins / self.total_games
    @property
    def white_win_rate(self) -> float: return self.white_wins / self.total_games
    @property
    def draw_rate(self)  -> float:     return self.draws      / self.total_games
    @property
    def seconds_per_move(self) -> float:
        return self.duration_seconds / max(1, self.total_moves)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.update({
            "black_win_rate":  self.black_win_rate,
            "white_win_rate":  self.white_win_rate,
            "draw_rate":       self.draw_rate,
            "seconds_per_move": self.seconds_per_move,
        })
        return d

# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------
class BatchGameSimulator:
    def __init__(self, cfg: SimulationConfig):
        self.cfg    = cfg
        self.device = cfg.device or self._select_device()

    # ----------------------- helpers ---------------------------------
    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    # ----------------------- public API ------------------------------
    def simulate(self) -> GameStatistics:
        print(f"Simulating {self.cfg.num_games} games on "
              f"{self.cfg.board_size}×{self.cfg.board_size} ({self.device})")

        boards = TensorBoard(self.cfg.num_games,
                             self.cfg.board_size,
                             self.device)
        bot = TensorBatchBot(device=self.device)

        t0 = time.time()
        with torch.no_grad():                       # ❶ no autograd
            total_moves = self._run_game_loop(boards, bot)
        dt = time.time() - t0

        stats = self._compute_stats(boards, total_moves, dt)
        self._display_results(stats)
        self._display_boards(boards)
        return stats

    # ----------------------- core loop -------------------------------
    def _run_game_loop(self, boards: TensorBoard, bot: TensorBatchBot) -> int:
        move      = 0
        finished  = boards.is_game_over()           # device tensor cache

        while move < self.cfg.max_moves and not finished.all():
            moves  = bot.select_moves(boards)
            boards.step(moves)

            finished |= boards.is_game_over()       # in-place OR
            move += 1
            self._log_progress(finished, move)

        return move

    # ----------------------- utilities -------------------------------
    def _log_progress(self, finished: Tensor, ply: int) -> None:
        if self.cfg.log_interval and ply % self.cfg.log_interval == 0:
            done = finished.sum().item()            # one sync per interval
            print(f"Move {ply:4d}: {done}/{self.cfg.num_games} finished")

    def _compute_stats(self, boards: TensorBoard,
                       mv: int, dt: float) -> GameStatistics:
        scores = boards.compute_scores().cpu()
        b, w   = scores[:, 0], scores[:, 1]
        bw     = (b > w).sum().item()
        ww     = (w > b).sum().item()
        dr     = self.cfg.num_games - bw - ww
        return GameStatistics(self.cfg.num_games, mv, dt, bw, ww, dr, scores)

    def _display_results(self, s: GameStatistics) -> None:
        print(f"\nFinished in {s.duration_seconds:.2f}s "
              f"(avg {s.seconds_per_move:.4f}s per ply)\n")
        print(f"Black wins : {s.black_wins:4d} ({s.black_win_rate:6.1%})")
        print(f"White wins : {s.white_wins:4d} ({s.white_win_rate:6.1%})")
        print(f"Draws      : {s.draws:4d} ({s.draw_rate:6.1%})\n")

    def _display_boards(self, boards: TensorBoard) -> None:
        if self.cfg.show_boards:
            n = min(self.cfg.show_boards, self.cfg.num_games)
            for i in range(n):
                show(boards, header=f"Game {i + 1}", idx=i)

# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------
def simulate_batch_games(num_games=100, board_size=19, **kw) -> GameStatistics:
    return BatchGameSimulator(SimulationConfig(num_games, board_size, **kw)).simulate()

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def main():
    simulate_batch_games(num_games=512,
                         board_size=8,
                         show_boards=2,
                         log_interval=64)

if __name__ == "__main__":
    main()

