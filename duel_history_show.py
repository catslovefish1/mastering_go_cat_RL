"""
tensor_batch_duel.py — batch self-play driver (optimised, no profiler)

Run:
    python tensor_batch_duel.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

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
    max_moves_factor: int = 10      # max plies = board² × factor
    show_boards: int = 0            # 0 → suppress pretty-printed boards
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
    final_scores: Tensor            # (B, 2)

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
            "black_win_rate":   self.black_win_rate,
            "white_win_rate":   self.white_win_rate,
            "draw_rate":        self.draw_rate,
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

        # ==== NEW ----------------------------------------------------
        # keep the complete move list for *game 0* (the first board)
        # elements: (colour, (row, col)) with 1-based coordinates
        self.first_game_history: List[Tuple[str, Tuple[int, int]]] = []
        # -------------------------------------------------------------

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
        self._display_history()          # ==== NEW ====
        self._display_boards(boards)     # (optional pretty boards)
        return stats

    # ----------------------- core loop -------------------------------
    def _run_game_loop(self, boards: TensorBoard, bot: TensorBatchBot) -> int:
        ply       = 0                                 # number of plies so far
        finished  = boards.is_game_over()             # device tensor cache

        while ply < self.cfg.max_moves and not finished.all():
            moves = bot.select_moves(boards)          # (B,) tensor of indices
            boards.step(moves)

            # ==== NEW ------------------------------------------------
            # remember the move for the first board only (index 0)
            mv_idx = moves[0].item()                  # integer in 0 … N²-1 (or -1/pass)
            if mv_idx >= 0:                           # ignore passes for now
                colour = 'B' if (ply % 2 == 0) else 'W'
                N      = boards.size                  # board dimension
                row    = mv_idx // N + 1              # 1-based for humans
                col    = mv_idx %  N + 1
                self.first_game_history.append((colour, (row, col)))
            else:
                colour = 'B' if (ply % 2 == 0) else 'W'
                self.first_game_history.append((colour, ('pass',)))
            # ---------------------------------------------------------

            finished |= boards.is_game_over()         # in-place OR
            ply += 1
            self._log_progress(finished, ply)

        return ply

    # ----------------------- utilities -------------------------------
    def _log_progress(self, finished: Tensor, ply: int) -> None:
        if self.cfg.log_interval and ply % self.cfg.log_interval == 0:
            done = finished.sum().item()              # one sync per interval
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

    # ==== NEW --------------------------------------------------------
    def _display_history(self) -> None:
        """Pretty-print the complete move list for game #1."""
        if not self.first_game_history:
            return
        print("Move history — Game 1")
        for k, (clr, pt) in enumerate(self.first_game_history, start=1):
            if pt == ('pass',):
                print(f"Step {k:3d}: {clr}  pass")
            else:
                r, c = pt
                print(f"Step {k:3d}: {clr}  ({r:2d}, {c:2d})")
        print()  # blank line after history
    # ---------------------------------------------------------------

    def _display_boards(self, boards: TensorBoard) -> None:
        """Original per-game ASCII boards (kept optional)."""
        if self.cfg.show_boards:
            n = min(self.cfg.show_boards, self.cfg.num_games)
            for i in range(n):
                show(boards, header=f"Game {i + 1}", idx=i)

# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------
def simulate_batch_games(num_games=100, board_size=19, **kw) -> GameStatistics:
    return BatchGameSimulator(
        SimulationConfig(num_games, board_size, **kw)
    ).simulate()

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def main():
    simulate_batch_games(
        num_games=2**10,
        board_size=19,
        show_boards=0,      # keep ASCII boards silent; only history prints
        log_interval=64,
    )

if __name__ == "__main__":
    main()
