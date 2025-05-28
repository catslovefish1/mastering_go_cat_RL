"""
enhanced_profiling_simulator.py - Enhanced profiling for Go engine performance analysis
--------------------------------------------------------------------------------------

This version adds:
1. Detailed operation-level profiling
2. Memory profiling
3. Multiple output formats (JSON trace, text report, CSV)
4. Automatic bottleneck detection
5. Per-operation statistics

Usage:
    python enhanced_profiling_simulator.py
    
Outputs:
    - trace.json: Chrome tracing format (view at chrome://tracing)
    - profile_report.txt: Human-readable text report
    - profile_stats.csv: CSV for further analysis
    - memory_snapshot.pickle: Memory profiling data
"""

from __future__ import annotations
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pickle
import csv

import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from torch import Tensor

# Import your TensorBoard and bot
from engine.tensor_native import TensorBoard
from agents.basic import TensorBatchBot
from interface.ascii import show

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class ProfilingConfig:
    # Game settings
    num_games: int = 100
    board_size: int = 19
    max_moves_factor: float = 2.0
    
    # Display settings
    show_boards: int = 0
    log_interval: int = 100
    
    # Profiling settings
    profile_memory: bool = True
    profile_stack: bool = True
    profile_modules: bool = True
    export_chrome_trace: bool = True
    export_text_report: bool = True
    export_csv_stats: bool = True
    
    # Profiler schedule
    wait_steps: int = 1
    warmup_steps: int = 3
    active_steps: int = 10
    repeat: int = 1
    
    device: Optional[torch.device] = None
    output_dir: str = "profiling_results"

    @property
    def max_moves(self) -> int:
        return int(self.board_size * self.board_size * self.max_moves_factor)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Enhanced Profiling Simulator
# ----------------------------------------------------------------------
class EnhancedProfilingSimulator:
    def __init__(self, cfg: ProfilingConfig):
        self.cfg = cfg
        self.device = cfg.device or self._select_device()
        self.profiling_results = {}
        
    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def simulate(self) -> Dict[str, Any]:
        """Run simulation with comprehensive profiling"""
        print(f"Simulating {self.cfg.num_games} games on "
              f"{self.cfg.board_size}×{self.cfg.board_size} ({self.device})")
        print(f"Profiling output will be saved to: {self.cfg.output_dir}/")
        
        # Create board and bot
        boards = TensorBoard(self.cfg.num_games, self.cfg.board_size, self.device)
        bot = TensorBatchBot(device=self.device)
        
        # Run profiled simulation
        t0 = time.time()
        total_moves = self._run_profiled_game_loop(boards, bot)
        duration = time.time() - t0
        
        # Compute game statistics
        stats = self._compute_game_stats(boards, total_moves, duration)
        
        # Display results
        self._display_results(stats)
        self._display_boards(boards)
        
        return {
            "game_stats": stats,
            "profiling_results": self.profiling_results
        }
    
    def _run_profiled_game_loop(self, boards: TensorBoard, bot: TensorBatchBot) -> int:
        """Run game loop with detailed profiling"""
        move = 0
        finished = boards.is_game_over()
        
        # Setup profiler
        prof_schedule = schedule(
            wait=self.cfg.wait_steps,
            warmup=self.cfg.warmup_steps,
            active=self.cfg.active_steps,
            repeat=self.cfg.repeat
        )
        
        activities = [ProfilerActivity.CPU]
        if self.device.type in ['cuda', 'mps']:
            activities.append(ProfilerActivity.CUDA)
        
        profile_kwargs = {
            "activities": activities,
            "schedule": prof_schedule,
            "record_shapes": True,
            "profile_memory": self.cfg.profile_memory,
            "with_stack": self.cfg.profile_stack,
            "with_modules": self.cfg.profile_modules,
        }
        
        def trace_handler(prof):
            # Export chrome trace
            if self.cfg.export_chrome_trace:
                trace_path = os.path.join(self.cfg.output_dir, "trace.json")
                prof.export_chrome_trace(trace_path)
                print(f"\nChrome trace saved to: {trace_path}")
            
            # Export text report
            if self.cfg.export_text_report:
                report_path = os.path.join(self.cfg.output_dir, "profile_report.txt")
                with open(report_path, 'w') as f:
                    # CPU time sorted
                    f.write("=" * 80 + "\n")
                    f.write("PROFILE REPORT - CPU TIME\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(prof.key_averages().table(
                        sort_by="cpu_time_total", row_limit=50
                    ))
                    
                    # CUDA time sorted (if available)
                    if self.device.type in ['cuda', 'mps']:
                        f.write("\n\n" + "=" * 80 + "\n")
                        f.write("PROFILE REPORT - CUDA TIME\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(prof.key_averages().table(
                            sort_by="cuda_time_total", row_limit=50
                        ))
                    
                    # Memory sorted
                    if self.cfg.profile_memory:
                        f.write("\n\n" + "=" * 80 + "\n")
                        f.write("PROFILE REPORT - MEMORY\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(prof.key_averages().table(
                            sort_by="self_cpu_memory_usage", row_limit=50
                        ))
                
                print(f"Text report saved to: {report_path}")
            
            # Export CSV stats
            if self.cfg.export_csv_stats:
                self._export_csv_stats(prof)
            
            # Store key metrics
            self._extract_key_metrics(prof)
        
        profile_kwargs["on_trace_ready"] = trace_handler
        
        # Main profiling loop
        with profile(**profile_kwargs) as prof:
            while move < self.cfg.max_moves:
                if finished.all():
                    break
                
                # Detailed profiling of each major operation
                with record_function("MOVE_ITERATION"):
                    # Legal move computation
                    with record_function("compute_legal_moves"):
                        legal_moves = boards.legal_moves()
                    
                    # Bot move selection
                    with record_function("bot_select_moves"):
                        moves = bot.select_moves(boards)
                    
                    # Board update
                    with record_function("board_step"):
                        boards.step(moves)
                    
                    # Game over check
                    with record_function("check_game_over"):
                        new_finished = boards.is_game_over()
                        finished |= new_finished
                
                move += 1
                prof.step()
                
                self._log_progress(finished, move)
        
        # Memory snapshot if requested
        if self.cfg.profile_memory and self.device.type == 'cuda':
            self._capture_memory_snapshot()
        
        return move
    
    def _export_csv_stats(self, prof):
        """Export profiling statistics to CSV"""
        csv_path = os.path.join(self.cfg.output_dir, "profile_stats.csv")
        
        stats = []
        for item in prof.key_averages():
            stats.append({
                "name": item.key,
                "cpu_time_total_ms": item.cpu_time_total / 1000,
                "cuda_time_total_ms": item.cuda_time_total / 1000 if hasattr(item, 'cuda_time_total') else 0,
                "cpu_time_avg_ms": item.cpu_time / 1000,
                "cuda_time_avg_ms": item.cuda_time / 1000 if hasattr(item, 'cuda_time') else 0,
                "count": item.count,
                "self_cpu_memory_mb": item.self_cpu_memory_usage / (1024 * 1024) if hasattr(item, 'self_cpu_memory_usage') else 0,
            })
        
        # Sort by total CPU time
        stats.sort(key=lambda x: x["cpu_time_total_ms"], reverse=True)
        
        with open(csv_path, 'w', newline='') as f:
            if stats:
                writer = csv.DictWriter(f, fieldnames=stats[0].keys())
                writer.writeheader()
                writer.writerows(stats)
        
        print(f"CSV stats saved to: {csv_path}")
    
    def _extract_key_metrics(self, prof):
        """Extract key performance metrics from profiler"""
        key_ops = [
            "compute_legal_moves",
            "bot_select_moves", 
            "board_step",
            "_flood_fill",
            "_count_neighbors",
            "_process_captures"
        ]
        
        metrics = {}
        for item in prof.key_averages():
            for op in key_ops:
                if op in item.key:
                    metrics[op] = {
                        "cpu_time_ms": item.cpu_time_total / 1000,
                        "cuda_time_ms": item.cuda_time_total / 1000 if hasattr(item, 'cuda_time_total') else 0,
                        "count": item.count,
                        "avg_time_ms": item.cpu_time / 1000,
                    }
        
        self.profiling_results["key_metrics"] = metrics
        
        # Print summary
        print("\n" + "=" * 60)
        print("KEY OPERATION METRICS")
        print("=" * 60)
        for op, data in sorted(metrics.items(), key=lambda x: x[1]["cpu_time_ms"], reverse=True):
            print(f"\n{op}:")
            print(f"  Total CPU time: {data['cpu_time_ms']:.2f} ms")
            if data['cuda_time_ms'] > 0:
                print(f"  Total CUDA time: {data['cuda_time_ms']:.2f} ms")
            print(f"  Count: {data['count']}")
            print(f"  Avg time per call: {data['avg_time_ms']:.3f} ms")
    
    def _capture_memory_snapshot(self):
        """Capture detailed memory snapshot"""
        if self.device.type == 'cuda':
            snapshot = torch.cuda.memory_snapshot()
            snapshot_path = os.path.join(self.cfg.output_dir, "memory_snapshot.pickle")
            with open(snapshot_path, 'wb') as f:
                pickle.dump(snapshot, f)
            print(f"Memory snapshot saved to: {snapshot_path}")
    
    def _log_progress(self, finished: Tensor, move_no: int) -> None:
        if self.cfg.log_interval and move_no % self.cfg.log_interval == 0:
            print(f"Move {move_no:4d}: {finished.sum().item()}/"
                  f"{self.cfg.num_games} finished")
    
    def _compute_game_stats(self, boards: TensorBoard, total_moves: int, duration: float) -> Dict[str, Any]:
        """Compute game statistics"""
        scores = boards.compute_scores().cpu()
        black_scores, white_scores = scores[:, 0], scores[:, 1]
        
        black_wins = (black_scores > white_scores).sum().item()
        white_wins = (white_scores > black_scores).sum().item()
        draws = self.cfg.num_games - black_wins - white_wins
        
        return {
            "total_games": self.cfg.num_games,
            "total_moves": total_moves,
            "duration_seconds": duration,
            "black_wins": black_wins,
            "white_wins": white_wins,
            "draws": draws,
            "black_win_rate": black_wins / self.cfg.num_games,
            "white_win_rate": white_wins / self.cfg.num_games,
            "draw_rate": draws / self.cfg.num_games,
            "moves_per_second": total_moves / duration,
            "games_per_second": self.cfg.num_games / duration,
        }
    
    def _display_results(self, stats: Dict[str, Any]) -> None:
        """Display game results"""
        print(f"\nFinished in {stats['duration_seconds']:.2f}s")
        print(f"Total moves: {stats['total_moves']}")
        print(f"Moves per second: {stats['moves_per_second']:.0f}")
        print(f"Games per second: {stats['games_per_second']:.1f}\n")
        
        print(f"Black wins: {stats['black_wins']:4d} ({stats['black_win_rate']:6.1%})")
        print(f"White wins: {stats['white_wins']:4d} ({stats['white_win_rate']:6.1%})")
        print(f"Draws:      {stats['draws']:4d} ({stats['draw_rate']:6.1%})")
    
    def _display_boards(self, boards: TensorBoard) -> None:
        """Display sample boards"""
        if self.cfg.show_boards:
            n = min(self.cfg.show_boards, self.cfg.num_games)
            for i in range(n):
                show(boards, header=f"Game {i+1}", idx=i)

# ----------------------------------------------------------------------
# Analysis Helper Functions
# ----------------------------------------------------------------------

def analyze_profile_results(output_dir: str = "profiling_results"):
    """Analyze and summarize profiling results"""
    print("\n" + "=" * 80)
    print("PROFILING ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Read the text report
    report_path = os.path.join(output_dir, "profile_report.txt")
    if os.path.exists(report_path):
        print(f"\nText report available at: {report_path}")
        # Extract top operations
        with open(report_path, 'r') as f:
            lines = f.readlines()
            in_table = False
            op_count = 0
            print("\nTop 5 most expensive operations:")
            for line in lines:
                if "Name" in line and "CPU time" in line:
                    in_table = True
                    continue
                if in_table and line.strip() and op_count < 5:
                    if not line.startswith("-"):
                        print(f"  {line.strip()}")
                        op_count += 1
    
    # Read CSV stats
    csv_path = os.path.join(output_dir, "profile_stats.csv")
    if os.path.exists(csv_path):
        total_time = 0
        key_ops_time = {}
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_time += float(row['cpu_time_total_ms'])
                
                # Track specific operations
                for op in ['flood_fill', 'count_neighbors', 'legal_moves', 'step']:
                    if op in row['name']:
                        if op not in key_ops_time:
                            key_ops_time[op] = 0
                        key_ops_time[op] += float(row['cpu_time_total_ms'])
        
        print(f"\nTotal profiled CPU time: {total_time:.2f} ms")
        print("\nTime breakdown for key operations:")
        for op, time_ms in sorted(key_ops_time.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_ms / total_time) * 100 if total_time > 0 else 0
            print(f"  {op}: {time_ms:.2f} ms ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def main():
    """Run enhanced profiling simulation"""
    config = ProfilingConfig(
        num_games=512,
        board_size=8,  # Smaller for faster profiling
        show_boards=2,
        log_interval=64,
        
        # Profiling settings
        profile_memory=True,
        profile_stack=True,
        export_chrome_trace=True,
        export_text_report=True,
        export_csv_stats=True,
        
        # Adjust profiler schedule
        wait_steps=2,
        warmup_steps=5,
        active_steps=20,
    )
    
    simulator = EnhancedProfilingSimulator(config)
    results = simulator.simulate()
    
    # Analyze results
    analyze_profile_results(config.output_dir)
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.output_dir}/")
    print("\nTo view results:")
    print("1. Chrome trace: Open chrome://tracing and load trace.json")
    print("2. Text report: Open profile_report.txt")
    print("3. CSV stats: Open profile_stats.csv in Excel/spreadsheet")
    print("4. TensorBoard: tensorboard --logdir=profiling_results")
    
    return results

if __name__ == "__main__":
    main()