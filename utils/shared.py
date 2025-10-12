"""utils/shared.py - Shared utilities for tensor-based Go implementation.

This module contains common utilities used across the Go engine, agents, and simulation.
All utilities are self-contained and can be imported by any module.
"""

from __future__ import annotations
import time
from collections import defaultdict
from functools import wraps
from typing import Tuple, Optional, TYPE_CHECKING, Dict, List, Callable

import torch
from torch import Tensor

# Avoid circular imports
if TYPE_CHECKING:
    from engine.tensor_native_stable_dense import TensorBoard

# ========================= DEVICE UTILITIES =========================

def select_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU.
    
    Returns:
        torch.device: The best available device for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ========================= COORDINATE UTILITIES =========================

def flat_to_2d(flat_indices: Tensor, width: int) -> Tuple[Tensor, Tensor]:
    """Convert flat indices to 2D coordinates.
    
    Args:
        flat_indices: Shape (N,) flat position indices
        width: Board width for modulo operation
        
    Returns:
        Tuple of (rows, cols) tensors
    """
    rows = flat_indices // width
    cols = flat_indices % width
    return rows, cols

def coords_to_flat(rows: Tensor, cols: Tensor, width: int) -> Tensor:
    """Convert 2D coordinates to flat indices.
    
    Args:
        rows: Row coordinates
        cols: Column coordinates
        width: Board width
        
    Returns:
        Flat indices tensor
    """
    return rows * width + cols



# ========================= PROBABILITY UTILITIES =========================

def compute_uniform_probabilities(mask: Tensor) -> Tensor:
    """Compute uniform probability distribution over True values in mask.
    
    Args:
        mask: Shape (N, M) boolean mask
        
    Returns:
        Shape (N, M) probability distribution (sums to 1 along dim=1)
    """
    # Convert to float for probability computation
    probabilities = mask.float()
    
    # Normalize each row to sum to 1
    # keepdim=True maintains shape for broadcasting
    row_sums = probabilities.sum(dim=1, keepdim=True)
    safe_sums = row_sums.clamp(min=1.0)  # Avoid division by zero
    probabilities = probabilities / safe_sums
    
    return probabilities

def sample_from_mask(mask: Tensor, num_samples: int = 1) -> Tensor:
    """Sample indices from a boolean mask with uniform probability.
    
    Args:
        mask: Shape (N, M) boolean mask
        num_samples: Number of samples per row
        
    Returns:
        Shape (N, num_samples) or (N,) if num_samples=1
    """
    probabilities = compute_uniform_probabilities(mask)
    sampled = torch.multinomial(probabilities, num_samples=num_samples)
    
    if num_samples == 1:
        sampled = sampled.squeeze(1)
    
    return sampled



# ========================= BATCH UTILITIES =========================



# ========================= TIMING UTILITIES =========================

class TimingContext:
    """Context manager for timing operations with MPS synchronization"""
    def __init__(self, timer_dict: Dict[str, List[float]], name: str, device: torch.device, enable_timing: bool = True):
        self.timer_dict = timer_dict
        self.name = name
        self.device = device
        self.start_time = None
        self.enable_timing = enable_timing
    
    def __enter__(self):
        if not self.enable_timing:
            return self
        if self.device.type == 'mps':
            torch.mps.synchronize()  # Ensure previous operations are complete
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable_timing:
            return
        if self.device.type == 'mps':
            torch.mps.synchronize()  # Ensure current operation is complete
        elapsed = time.perf_counter() - self.start_time
        self.timer_dict[self.name].append(elapsed)

def timed_method(method: Callable) -> Callable:
    """Decorator to time methods with MPS synchronization"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Check if timing is enabled
        if hasattr(self, 'enable_timing') and not self.enable_timing:
            return method(self, *args, **kwargs)
        
        # Use TimingContext if the object has the required attributes
        if hasattr(self, 'timings') and hasattr(self, 'device'):
            # Increment call count if available
            if hasattr(self, 'call_counts'):
                self.call_counts[method.__name__] += 1
            
            with TimingContext(self.timings, method.__name__, self.device, getattr(self, 'enable_timing', True)):
                return method(self, *args, **kwargs)
        else:
            # Fallback: just run the method without timing
            return method(self, *args, **kwargs)
    
    return wrapper

# ========================= PRINTING UTILITIES =========================

def print_section_header(title: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)




def print_timing_report(board, top_n: int = 30) -> None:
    """Simplified timing report for board.timings."""
    import torch

    def human_bytes(n: float) -> str:
        n = abs(float(n))
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024.0
        return f"{n:.2f} PB"

    time_rows, mem_rows = [], []
    timings = getattr(board, "timings", {}) or {}

    for name, values in timings.items():
        if not values:
            continue
        t = torch.tensor(values, dtype=torch.float64)
        entry = {
            "name": name,
            "total": float(t.sum().item()),
            "mean": float(t.mean().item()),
            "count": int(t.numel()),
        }
        if ":mem" in name:
            mem_rows.append(entry)
        else:
            time_rows.append(entry)

    # Print timing table
    if time_rows:
        time_rows = sorted(time_rows, key=lambda r: r["total"], reverse=True)[:top_n]
        print("\n" + "="*80)
        print(f"{'Function':<44} {'Total(ms)':>11} {'Avg(ms)':>10} {'Calls':>8}")
        print("-"*80)
        for r in time_rows:
            print(f"{r['name']:<44} {r['total']*1000:>11.2f} {r['mean']*1000:>10.3f} {r['count']:>8}")

    # Print memory table
    if mem_rows:
        mem_rows = sorted(mem_rows, key=lambda r: r["total"], reverse=True)[:top_n]
        print("\n" + "="*80)
        print(f"{'Function':<44} {'Total':>13} {'Avg':>13} {'Calls':>8}")
        print("-"*80)
        for r in mem_rows:
            print(f"{r['name']:<44} {human_bytes(r['total']):>13} {human_bytes(r['mean']):>13} {r['count']:>8}")
    
 

def print_performance_metrics(elapsed: float, moves_made: int, num_games: int) -> None:
    """Print performance metrics for a simulation run."""
    print_section_header("PERFORMANCE METRICS")
    print(f"Total simulation time: {elapsed:.2f} seconds")
    print(f"Moves per second: {moves_made/elapsed:.1f}")
    print(f"Games per second: {num_games/elapsed:.1f}")
    print(f"Time per move: {elapsed/moves_made*1000:.2f} ms")
    print(f"Time per game: {elapsed/num_games:.3f} seconds")


import json, os, torch


def save_game_histories_to_json(
    boards,
    num_games_to_save: int = 5,
    output_dir: str = "game_histories",
) -> None:
    """
    Dump up to *num_games_to_save* finished games from a TensorBoard batch
    into individual JSON files.  Works with the refactored data structure:

        boards.board         – (B, H, W) int8   (-1 empty, 0 black, 1 white)
        boards.board_history – (B, T, N2) int8  snapshots taken *before* each move
        boards.hash_history  – (B, T)   int64   Zobrist hash of the same snapshots
        boards.current_hash  – (B,)     int64   hash of the *current* board

    Notes on hashes:
    - For move m (1-based in the JSON), we emit:
        pre_hash  = hash of the board before move m  (hash_history[g, m-1])
        post_hash = hash of the board after  move m  (hash_history[g, m]
                                                      if exists, else current_hash[g])
    - This keeps hashes aligned with what board_history already stores (pre-move states).
    """
    os.makedirs(output_dir, exist_ok=True)

    B          = boards.batch_size
    max_moves  = boards.board_history.shape[1]
    N          = boards.board_size
    board_area = N * N
    dev        = boards.device

    # Optional (present only if super-ko was enabled)
    has_hash_hist = hasattr(boards, "hash_history") and boards.hash_history is not None
    has_curr_hash = hasattr(boards, "current_hash") and boards.current_hash is not None

    # Convenience constant for an "all empty" board (pre-previous state)
    EMPTY = torch.full((board_area,), -1, dtype=torch.int8, device=dev)

    for g in range(min(num_games_to_save, B)):
        total_moves = int(boards.move_count[g].item()) if hasattr(boards.move_count, "device") else int(boards.move_count[g])
        recorded    = min(total_moves, max_moves)

        prev = EMPTY
        moves = []

        for m in range(recorded):
            # Pre-move board snapshot at ply m (0-based)
            curr = boards.board_history[g, m]                       # (N2,) int8

            # Per-move hashes (pre/post)
            pre_hash  = int(boards.hash_history[g, m].item()) if has_hash_hist else None
            if has_hash_hist:
                # post_hash is the next pre-snapshot if it exists, else current board hash
                if m + 1 < max_moves and (m + 1) < recorded:
                    post_hash = int(boards.hash_history[g, m + 1].item())
                else:
                    post_hash = int(boards.current_hash[g].item()) if has_curr_hash else None
            else:
                post_hash = None

            # Diff vs previous snapshot → changes introduced by the *last* move
            diff = (curr != prev).nonzero(as_tuple=True)[0]

            move_descr = {
                "move_number": m + 1,
                "board_state": curr.detach().cpu().tolist(),
                "changes": [],
                "pre_hash":  pre_hash,
                "post_hash": post_hash,
            }

            if diff.numel() == 0:     # pure pass
                move_descr["changes"].append({"type": "pass"})
            else:
                for pos in diff.tolist():
                    row, col = divmod(int(pos), N)
                    old = int(prev[pos].item())
                    new = int(curr[pos].item())

                    if old == -1 and new != -1:        # placement
                        move_descr["changes"].append({
                            "type": "place",
                            "position": [row, col],
                            "color": "black" if new == 0 else "white",
                        })
                    elif old != -1 and new == -1:      # capture
                        move_descr["changes"].append({
                            "type": "capture",
                            "position": [row, col],
                            "color": "black" if old == 0 else "white",
                        })
                    else:
                        # Rare but possible if something rewrites stones directly
                        move_descr["changes"].append({
                            "type": "flip",
                            "position": [row, col],
                            "from": "empty" if old == -1 else ("black" if old == 0 else "white"),
                            "to":   "empty" if new == -1 else ("black" if new == 0 else "white"),
                        })

            moves.append(move_descr)
            prev = curr.clone()

        # --- final score: just count stones on the board -----------------
        final_board = boards.board[g]           # (H, W) int8
        black_stones = int((final_board == 0).sum().item())
        white_stones = int((final_board == 1).sum().item())

        # Final hash (current board)
        final_hash = int(boards.current_hash[g].item()) if has_curr_hash else None

        game_json = {
            "game_id": g,
            "board_size": N,
            "total_moves": total_moves,
            "moves_recorded": recorded,
            "truncated": total_moves > max_moves,
            "final_score": {"black": black_stones, "white": white_stones},
            "final_hash": final_hash,
            "moves": moves,
        }

        with open(os.path.join(output_dir, f"game_{g:03d}.json"), "w") as f:
            json.dump(game_json, f, indent=2)
