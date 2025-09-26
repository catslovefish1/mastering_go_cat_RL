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
    from engine.tensor_native import TensorBoard

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

# ========================= POSITION UTILITIES =========================

def create_pass_positions(batch_size: int, device: torch.device) -> Tensor:
    """Create tensor of pass moves for given batch size.
    
    Pass moves are represented as [-1, -1] in the position tensor.
    
    Args:
        batch_size: Number of positions to create
        device: Target device for tensor creation
        
    Returns:
        Tensor of shape (batch_size, 2) filled with -1
    """
    return torch.full((batch_size, 2), -1, dtype=torch.int16, device=device)


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

def print_union_find_grid(board, batch_idx: int = 0, column: int = 0) -> None:
    """Print union-find data in grid format
    column: 0=Colour, 1=Parent, 2=Liberty
    """
    column_names = ["Colour", "Parent", "Liberty"]
    print(f"\n{column_names[column]} values for batch {batch_idx}:")
    print("-" * (board.board_size * 4 + 1))
    
    uf_data = board.flatten_union_find[batch_idx, :, column].view(board.board_size, board.board_size)
    
    for row in range(board.board_size):
        row_str = "|"
        for col in range(board.board_size):
            value = uf_data[row, col].item()
            row_str += f"{value:3}|"
        print(row_str)
    print("-" * (board.board_size * 4 + 1))

def print_all_union_find_columns(board, batch_idx: int = 0, board_size_limit: int = 9) -> None:
    """Print all union-find columns with appropriate formatting."""
    # Always print colour
    print("\nCOLOUR (-1=empty, 0=black, 1=white):")
    print_union_find_grid(board, batch_idx, column=0)
    
    # Only print parent/liberty for small boards
    if board.board_size <= board_size_limit:
        print("\nPARENT INDICES:")
        print_union_find_grid(board, batch_idx, column=1)
        
        print("\nLIBERTY COUNTS:")
        print_union_find_grid(board, batch_idx, column=2)

def print_move_info(move: Tensor, player: int) -> None:
    """Print information about a move."""
    player_name = "BLACK" if player == 0 else "WHITE"
    print(f"\nCurrent player: {player_name} ({player})")
    print(f"Move to be played: {move.tolist()}")
    
    if move[0] >= 0:  # Not a pass
        print(f"  Position: row={move[0].item()}, col={move[1].item()}")
    else:
        print("  PASS MOVE")

def print_game_state(board, batch_idx: int = 0, ply: int = 0, 
                    header: str = "", move: Optional[Tensor] = None) -> None:
    """Print complete game state for debugging.
    
    This is a high-level function that combines multiple printing utilities.
    """
    print_section_header(f"{header} - Ply {ply}")
    
    # Print move if provided
    if move is not None:
        print_move_info(move, board.current_player[batch_idx].item())
    
    # Print all union-find columns
    print_all_union_find_columns(board, batch_idx)
    
    # Additional game info
    print(f"\nPass count: {board.pass_count[batch_idx].item()}")
    if board.ko_points[batch_idx, 0] >= 0:
        ko_row = board.ko_points[batch_idx, 0].item()
        ko_col = board.ko_points[batch_idx, 1].item()
        print(f"Ko point: ({ko_row}, {ko_col})")

def print_timing_report(board, top_n: int = 30) -> None:
    """Print detailed timing report with enhanced statistics"""
    print_section_header("TIMING REPORT")
    
    # Convert to tensors for faster operations
    timing_data = []
    
    for func_name, times in board.timings.items():
        if times:
            times_tensor = torch.tensor(times, device='cpu')
            timing_data.append({
                'name': func_name,
                'times': times_tensor,
                'total': times_tensor.sum().item(),
                'mean': times_tensor.mean().item(),
                'std': times_tensor.std().item() if len(times) > 1 else 0,
                'count': len(times),
                'min': times_tensor.min().item(),
                'max': times_tensor.max().item(),
            })
    
    # Sort by total time
    timing_data.sort(key=lambda x: x['total'], reverse=True)
    
    # Calculate percentages
    total_time_all = sum(item['total'] for item in timing_data)
    
    # Print enhanced statistics
    print(f"\nTop {top_n} Time-Consuming Functions:")
    print(f"{'Function':<40} {'Total(ms)':<12} {'Avg(ms)':<12} {'Count':<10} {'%':<6}")
    print("-" * 80)
    
    for item in timing_data[:top_n]:
        percent = (item['total'] / total_time_all * 100) if total_time_all > 0 else 0
        print(f"{item['name']:<40} {item['total']*1000:<12.2f} {item['mean']*1000:<12.4f} "
              f"{item['count']:<10} {percent:<6.1f}")
    
    # Print call counts
    print(f"\n\nFunction Call Counts:")
    print(f"{'Function':<40} {'Calls':<10}")
    print("-" * 50)
    for func_name, count in sorted(board.call_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{func_name:<40} {count:<10}")
    
 

def print_performance_metrics(elapsed: float, moves_made: int, num_games: int) -> None:
    """Print performance metrics for a simulation run."""
    print_section_header("PERFORMANCE METRICS")
    print(f"Total simulation time: {elapsed:.2f} seconds")
    print(f"Moves per second: {moves_made/elapsed:.1f}")
    print(f"Games per second: {num_games/elapsed:.1f}")
    print(f"Time per move: {elapsed/moves_made*1000:.2f} ms")
    print(f"Time per game: {elapsed/num_games:.3f} seconds")

def print_game_summary(stats) -> None:
    """Print game statistics summary."""
    print(
        f"\nFinished {stats.total_games} games in {stats.duration_seconds:.2f}s "
        f"({stats.seconds_per_move:.4f}s/ply)"
    )
    print(f"Black wins: {stats.black_wins:4d} ({stats.black_win_rate:6.1%})")
    print(f"White wins: {stats.white_wins:4d} ({stats.white_win_rate:6.1%})")
    print(f"Draws     : {stats.draws:4d} ({stats.draw_rate:6.1%})")
    

# Add this to utils/shared.py

import json, os, torch

def save_game_histories_to_json(
    boards,
    num_games_to_save: int = 5,
    output_dir: str = "game_histories",
) -> None:
    """
    Dump up to *num_games_to_save* finished games from a TensorBoard batch
    into individual JSON files.  Works with the refactored data structure:

        boards.board  – (B, H, W) int8   (-1 = empty, 0 = black, 1 = white)
    """
    os.makedirs(output_dir, exist_ok=True)

    B          = boards.batch_size
    max_moves  = boards.board_history.shape[1]
    N          = boards.board_size
    board_area = N * N
    dev        = boards.device

    # Convenience constant for an "all empty" board
    EMPTY = torch.full((board_area,), -1, dtype=torch.int8, device=dev)

    for g in range(min(num_games_to_save, B)):
        total_moves = int(boards.move_count[g])
        recorded    = min(total_moves, max_moves)

        prev = EMPTY
        moves = []

        for m in range(recorded):
            curr = boards.board_history[g, m]
            diff = (curr != prev).nonzero(as_tuple=True)[0]

            move_descr = {
                "move_number": m + 1,
                "board_state": curr.cpu().tolist(),
                "changes": [],
            }

            if diff.numel() == 0:     # pure pass
                move_descr["changes"].append({"type": "pass"})
            else:
                for pos in diff:
                    row, col = divmod(int(pos), N)
                    old = int(prev[pos])
                    new = int(curr[pos])

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

            moves.append(move_descr)
            prev = curr.clone()

        # --- final score: just count stones on the board -----------------
        final_board = boards.board[g]           # (H, W) int8
        black_stones = int((final_board == 0).sum().item())
        white_stones = int((final_board == 1).sum().item())

        game_json = {
            "game_id": g,
            "board_size": N,
            "total_moves": total_moves,
            "moves_recorded": recorded,
            "truncated": total_moves > max_moves,
            "final_score": {"black": black_stones, "white": white_stones},
            "moves": moves,
        }

        with open(os.path.join(output_dir, f"game_{g:03d}.json"), "w") as f:
            json.dump(game_json, f, indent=2)
