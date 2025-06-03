"""tensor_native.py - Go engine with timing instrumentation.

Key improvements:
1. Uses centralized utilities from utils.shared
2. Cleaner separation of concerns
3. More functional programming style
4. ADDED: Comprehensive timing instrumentation for MPS performance analysis
"""

from __future__ import annotations
import os
import time
from typing import Tuple, Optional, Callable, Dict, List
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor

# Import shared utilities
from utils.shared import (
    select_device,
    flat_to_2d,
    coords_to_flat,
    create_pass_positions,
    is_pass_move,
    find_playable_games,
    get_batch_indices
)

# Environment setup
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ========================= CONSTANTS =========================

@dataclass(frozen=True)
class Stone:
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1

# Type aliases
BoardTensor = Tensor  # (B, H, W)
StoneTensor = Tensor  # (B, C, H, W)
PositionTensor = Tensor  # (B, 2)
BatchTensor = Tensor  # (B,)

# ========================= TIMING UTILITIES =========================

class TimingContext:
    """Context manager for timing operations with MPS synchronization"""
    def __init__(self, timer_dict: Dict[str, List[float]], name: str, device: torch.device):
        self.timer_dict = timer_dict
        self.name = name
        self.device = device
        self.start_time = None
    
    def __enter__(self):
        if self.device.type == 'mps':
            torch.mps.synchronize()  # Ensure previous operations are complete
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'mps':
            torch.mps.synchronize()  # Ensure current operation is complete
        elapsed = time.perf_counter() - self.start_time
        self.timer_dict[self.name].append(elapsed)

def timed_method(method: Callable) -> Callable:
    """Decorator to time methods with MPS synchronization"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with TimingContext(self.timings, method.__name__, self.device):
            return method(self, *args, **kwargs)
    return wrapper

# ========================= DECORATORS =========================

def with_active_games(method: Callable) -> Callable:
    """Decorator to handle active game filtering"""
    @wraps(method)
    def wrapper(self, *args, active_mask: Optional[BatchTensor] = None, **kwargs):
        if active_mask is None:
            active_mask = ~self.is_game_over()
        
        if not active_mask.any():
            return None
            
        return method(self, *args, active_mask=active_mask, **kwargs)
    return wrapper

# ========================= GO ENGINE =========================

class TensorBoard(torch.nn.Module):
    """Elegant vectorized Go board implementation with timing instrumentation"""
    
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        device: Optional[torch.device] = None,
        enable_timing: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.board_size = board_size
        self.device = device or select_device()
        self.enable_timing = enable_timing
        
        # Timing storage
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.flood_fill_iterations: List[int] = []  # Separate storage for iteration counts
        
        # CPU-GPU sync tracking
        self.sync_points: Dict[str, Dict] = {}  # Static analysis results
        self.sync_warnings: Dict[str, int] = defaultdict(int)  # Runtime warnings
        
        self._init_constants()
        self._init_state()
        self._cache = {}
    
    def _init_constants(self) -> None:
        """Initialize constant tensors"""
        with TimingContext(self.timings, '_init_constants', self.device):
            # Neighbor kernel for convolutions
            kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
                                dtype=torch.float32, device=self.device)
            self.register_buffer("neighbor_kernel", kernel.unsqueeze(0).unsqueeze(0))
            
            # Zobrist hashing
            torch.manual_seed(42)
            max_hash = torch.iinfo(torch.int64).max
            self.register_buffer(
                "zobrist_stones",
                torch.randint(0, max_hash, (2, self.board_size, self.board_size), 
                             dtype=torch.int64, device=self.device)
            )
            self.register_buffer(
                "zobrist_turn",
                torch.randint(0, max_hash, (2,), dtype=torch.int64, device=self.device)
            )
    
    def _init_state(self) -> None:
        """Initialize mutable game state"""
        with TimingContext(self.timings, '_init_state', self.device):
            B, H, W = self.batch_size, self.board_size, self.board_size
            
            self.register_buffer("stones", torch.zeros((B, 2, H, W), dtype=torch.bool, device=self.device))
            self.register_buffer("current_player", torch.zeros(B, dtype=torch.uint8, device=self.device))
            self.register_buffer("position_hash", torch.zeros(B, dtype=torch.int64, device=self.device))
            self.register_buffer("ko_points", torch.full((B, 2), -1, dtype=torch.int8, device=self.device))
            self.register_buffer("pass_count", torch.zeros(B, dtype=torch.uint8, device=self.device))
    
    # ==================== CORE UTILITIES ====================
    
    @timed_method
    def _count_neighbors(self, mask: BoardTensor) -> BoardTensor:
        """Count orthogonal neighbors"""
        self.call_counts['_count_neighbors'] += 1
        
        # Time each sub-operation
        with TimingContext(self.timings, '_count_neighbors.unsqueeze', self.device):
            mask_4d = mask.unsqueeze(1).float()
        
        with TimingContext(self.timings, '_count_neighbors.conv2d', self.device):
            counts = F.conv2d(mask_4d, self.neighbor_kernel, padding=1)
        
        with TimingContext(self.timings, '_count_neighbors.squeeze', self.device):
            result = counts.squeeze(1)
        
        return result
    
    @timed_method
    def _update_hash_for_positions(self, positions: Tuple[Tensor, Tensor, Tensor], colors: Tensor):
        """Update position hash for stone changes (unified method)"""
        batch_idx, rows, cols = positions
        flat_idx = coords_to_flat(rows, cols, self.board_size)
        hash_values = self.zobrist_stones[colors].view(colors.shape[0], -1)
        hash_values = hash_values.gather(1, flat_idx.unsqueeze(1)).squeeze(1)
        
        # XOR is its own inverse, so same operation for add/remove
        self.position_hash[batch_idx] ^= hash_values
    
    @with_active_games
    @timed_method
    def _toggle_turn(self, active_mask: BatchTensor) -> None:
        """Switch current player and update hash"""
        # Update hash for old player
        self.position_hash[active_mask] ^= self.zobrist_turn[self.current_player[active_mask].long()]
        # Switch player
        self.current_player[active_mask] ^= 1
        # Update hash for new player
        self.position_hash[active_mask] ^= self.zobrist_turn[self.current_player[active_mask].long()]
    
    def _invalidate_cache(self) -> None:
        """Clear cached values"""
        self._cache.clear()
    
    # ==================== BOARD QUERIES ====================
    
    @property
    def empty_mask(self) -> BoardTensor:
        """Get empty positions (cached)"""
        if 'empty' not in self._cache:
            with TimingContext(self.timings, 'empty_mask.compute', self.device):
                occupied = self.stones.any(dim=1)
                self._cache['empty'] = ~occupied
        return self._cache['empty']
    
    @timed_method
    def get_player_stones(self, player: Optional[int] = None) -> BoardTensor:
        """Get stones for specified player (None = current)"""
        if player is None:
            batch_idx = get_batch_indices(self.batch_size, self.device)
            return self.stones[batch_idx, self.current_player.long()]
        return self.stones[:, player]
    
    @timed_method
    def get_opponent_stones(self) -> BoardTensor:
        """Get opponent's stones"""
        batch_idx = get_batch_indices(self.batch_size, self.device)
        opponent = 1 - self.current_player
        return self.stones[batch_idx, opponent.long()]
    
    def get_playable_games(self, legal_moves: Optional[BoardTensor] = None) -> BatchTensor:
        """Get mask of games that have at least one legal move."""
        if legal_moves is None:
            legal_moves = self.legal_moves()
        
        return find_playable_games(legal_moves)
    
    # ==================== KO HANDLING ====================
    
    @timed_method
    def _apply_ko_restrictions(self, legal: BoardTensor) -> BoardTensor:
        """Apply ko restrictions to legal moves"""
        has_ko = self.ko_points[:, 0] >= 0
        if not has_ko.any():
            return legal
        
        # Remove ko points from legal moves
        ko_games = has_ko.nonzero(as_tuple=True)[0]
        ko_rows = self.ko_points[ko_games, 0].long()
        ko_cols = self.ko_points[ko_games, 1].long()
        
        legal = legal.clone()
        legal[ko_games, ko_rows, ko_cols] = False
        return legal
    
    @timed_method
    def _detect_ko(self, captured_positions: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Detect and set ko points for single stone captures"""
        batch_idx, rows, cols = captured_positions
        
        if batch_idx.numel() == 0:
            return
        
        # Count captures per batch using bincount
        capture_counts = torch.bincount(batch_idx, minlength=self.batch_size)
        
        # Find batches with exactly one capture
        single_capture_batches = (capture_counts == 1)
        
        if not single_capture_batches.any():
            return
        
        # For batches with single captures, find index
        batch_to_idx = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        idx_range = torch.arange(len(batch_idx), device=self.device)
        batch_to_idx.scatter_(0, batch_idx, idx_range)
        
        # Extract positions for single capture batches
        single_capture_idx = batch_to_idx[single_capture_batches]
        
        # Update ko points
        self.ko_points[single_capture_batches, 0] = rows[single_capture_idx].to(torch.int8)
        self.ko_points[single_capture_batches, 1] = cols[single_capture_idx].to(torch.int8)
    
    # ==================== LEGAL MOVES ====================
    
    @timed_method
    def legal_moves(self) -> BoardTensor:
        """Compute legal moves for current player"""
        self.call_counts['legal_moves'] += 1
        
        if 'legal' in self._cache:
            return self._cache['legal']
        
        # Get active (non-finished) games
        with TimingContext(self.timings, 'legal_moves.active_games', self.device):
            active_games = ~self.is_game_over()
        
        # No moves if all games are over
        if not active_games.any():
            legal = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                              dtype=torch.bool, device=self.device)
            self._cache['legal'] = legal
            return legal
        
        # Start with empty positions that have liberties
        with TimingContext(self.timings, 'legal_moves.basic_legal', self.device):
            legal = self.empty_mask & (self._count_neighbors(self.empty_mask) > 0)
        
        # Add capture moves
        with TimingContext(self.timings, 'legal_moves.capture_moves', self.device):
            legal |= self._find_capture_moves()
        
        # Apply ko restrictions
        with TimingContext(self.timings, 'legal_moves.ko_restrictions', self.device):
            legal = self._apply_ko_restrictions(legal)
        
        # Mask finished games
        legal[~active_games] = False
        
        self._cache['legal'] = legal
        return legal
    
    @timed_method
    def _find_capture_moves(self) -> BoardTensor:
        """Find moves that would capture opponent stones"""
        opponent = self.get_opponent_stones()
        
        # Find opponent stones with exactly 1 liberty
        liberties = self._count_neighbors(self.empty_mask) * opponent
        vulnerable = opponent & (liberties == 1)
        
        # Find empty points adjacent to vulnerable stones
        return self.empty_mask & (self._count_neighbors(vulnerable) > 0)
    
    # ==================== MOVE EXECUTION ====================
    
    @timed_method
    def step(self, positions: PositionTensor) -> None:
        """Execute moves for all games"""
        self._invalidate_cache()
        
        # Classify moves
        with TimingContext(self.timings, 'step.classify_moves', self.device):
            is_pass = is_pass_move(positions)
            finished = self.is_game_over()
            is_play = ~is_pass & ~finished
        
        # Update pass counter
        with TimingContext(self.timings, 'step.update_pass', self.device):
            self.pass_count = torch.where(
                is_pass | finished,
                torch.where(is_pass, self.pass_count + 1, self.pass_count),
                torch.zeros_like(self.pass_count)
            )
        
        # Reset ko
        with TimingContext(self.timings, 'step.reset_ko', self.device):
            if not finished.all():
                self.ko_points[~finished] = -1
        
        # Execute moves
        with TimingContext(self.timings, 'step.place_stones_check', self.device):
            if is_play.any():
                self._place_stones(positions[is_play], is_play.nonzero(as_tuple=True)[0])
        
        # Switch turn
        self._toggle_turn()
    
    @timed_method
    def _place_stones(self, positions: PositionTensor, batch_idx: Tensor) -> None:
        """Place stones and handle captures"""
        rows, cols = positions[:, 0].long(), positions[:, 1].long()
        colors = self.current_player[batch_idx].long()
        
        # Place stones
        with TimingContext(self.timings, '_place_stones.set_stones', self.device):
            self.stones[batch_idx, colors, rows, cols] = True
        
        # Update hash
        self._update_hash_for_positions((batch_idx, rows, cols), colors)
        
        # Handle captures
        self._process_captures(batch_idx, rows, cols)
    
    @timed_method
    def _process_captures(self, batch_idx: Tensor, rows: Tensor, cols: Tensor) -> None:
        """Remove captured opponent groups"""
        self.call_counts['_process_captures'] += 1
        
        # Find adjacent opponent stones
        with TimingContext(self.timings, '_process_captures.setup', self.device):
            move_mask = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                                   dtype=torch.float32, device=self.device)
            move_mask[batch_idx, rows, cols] = 1.0
        
        with TimingContext(self.timings, '_process_captures.find_neighbors', self.device):
            neighbors = self._count_neighbors(move_mask) > 0
            opponent_stones = self.get_opponent_stones()
            seeds = neighbors & opponent_stones
        
        if not seeds.any():
            return
        
        # Flood fill to find complete groups
        with TimingContext(self.timings, '_process_captures.flood_fill', self.device):
            groups = self._flood_fill(seeds, opponent_stones)
        
        # Find captured groups
        with TimingContext(self.timings, '_process_captures.find_captured', self.device):
            group_liberties = self._count_neighbors(self.empty_mask) * groups
            captured = groups & (group_liberties == 0)
        
        if captured.any():
            self._remove_captured_stones(captured)
    
    @timed_method
    def _flood_fill(self, seeds: BoardTensor, mask: BoardTensor) -> BoardTensor:
        """Expand seeds to complete groups"""
        self.call_counts['_flood_fill'] += 1
        groups = seeds.clone()
        iterations = 0
        
        while True:
            with TimingContext(self.timings, f'_flood_fill.iter', self.device):
                expanded = (self._count_neighbors(groups) > 0) & mask & ~groups
                if not expanded.any():
                    break
                groups |= expanded
                iterations += 1
        
        # Store iteration count separately (not as timing)
        self.flood_fill_iterations.append(iterations)
        return groups
    
    @timed_method
    def _remove_captured_stones(self, captured: BoardTensor) -> None:
        """Remove captured stones and handle ko"""
        positions = captured.nonzero(as_tuple=False)
        batch_idx = positions[:, 0]
        rows = positions[:, 1]
        cols = positions[:, 2]
        
        # Determine colors and remove stones
        colors = (1 - self.current_player[batch_idx]).long()
        self.stones[batch_idx, colors, rows, cols] = False
        
        # Update hash
        self._update_hash_for_positions((batch_idx, rows, cols), colors)
        
        # Detect ko
        self._detect_ko((batch_idx, rows, cols))
    
    # ==================== GAME STATE ====================
    
    def is_game_over(self) -> BatchTensor:
        """Check if games are finished"""
        return self.pass_count >= 2
    
    def compute_scores(self) -> Tensor:
        """Compute current scores"""
        black = self.stones[:, Stone.BLACK].sum(dim=(1, 2)).float()
        white = self.stones[:, Stone.WHITE].sum(dim=(1, 2)).float()
        return torch.stack([black, white], dim=1)
    
    
    def print_timing_report(self, top_n: int = 20) -> None:
        """Print detailed timing report"""
        print("\n" + "="*80)
        print("TIMING REPORT")
        print("="*80)
        
        # Aggregate timings
        timing_summary = []
        for func_name, times in self.timings.items():
            if times:
                total_time = sum(times)
                avg_time = total_time / len(times)
                count = len(times)
                timing_summary.append({
                    'function': func_name,
                    'total_ms': total_time * 1000,
                    'avg_ms': avg_time * 1000,
                    'count': count,
                    'percent': 0  # Will calculate after
                })
        
        # Calculate percentages
        total_time_all = sum(item['total_ms'] for item in timing_summary)
        for item in timing_summary:
            item['percent'] = (item['total_ms'] / total_time_all * 100) if total_time_all > 0 else 0
        
        # Sort by total time
        timing_summary.sort(key=lambda x: x['total_ms'], reverse=True)
        
        # Print top functions
        print(f"\nTop {top_n} Time-Consuming Functions:")
        print(f"{'Function':<40} {'Total (ms)':<12} {'Avg (ms)':<12} {'Count':<10} {'%':<6}")
        print("-" * 80)
        
        for item in timing_summary[:top_n]:
            print(f"{item['function']:<40} {item['total_ms']:<12.2f} {item['avg_ms']:<12.4f} "
                  f"{item['count']:<10} {item['percent']:<6.1f}")
        
        # Print call counts
        print(f"\n\nFunction Call Counts:")
        print(f"{'Function':<40} {'Calls':<10}")
        print("-" * 50)
        for func_name, count in sorted(self.call_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{func_name:<40} {count:<10}")
        
        # Special statistics for flood fill iterations
        if self.flood_fill_iterations:
            print(f"\n\nFlood Fill Statistics:")
            print(f"  Total flood fills: {len(self.flood_fill_iterations)}")
            print(f"  Average iterations: {sum(self.flood_fill_iterations)/len(self.flood_fill_iterations):.2f}")
            print(f"  Max iterations: {max(self.flood_fill_iterations)}")
            print(f"  Min iterations: {min(self.flood_fill_iterations)}")
            print(f"  Total iterations: {sum(self.flood_fill_iterations)}")
        
        # CPU-GPU Sync Analysis
        print("\n" + "="*80)
        print("CPU-GPU SYNCHRONIZATION ANALYSIS")
        print("="*80)
        
        # Analyze code for sync patterns
        sync_patterns = self._analyze_sync_patterns()
        
        if sync_patterns:
            print("\nPotential CPU-GPU Sync Points:")
            print(f"{'Function':<30} {'Sync Type':<20} {'Line/Pattern':<40}")
            print("-" * 90)
            
            for func_name, patterns in sync_patterns.items():
                for pattern in patterns:
                    print(f"{func_name:<30} {pattern['type']:<20} {pattern['pattern']:<40}")
        
        print("\nSync-Safe Operations Used:")
        safe_ops = [
            "✓ .any() for boolean checks (instead of .item())",
            "✓ .nonzero() returns GPU tensor",
            "✓ Mask-based operations stay on GPU",
            "✓ No Python loops over tensor elements"
        ]
        for op in safe_ops:
            print(f"  {op}")
        
        print("\n" + "="*80)

        """Analyze methods for CPU-GPU sync patterns"""
        sync_patterns = {}
        
        # Known sync points in current code
        problem_methods = {
            '_detect_ko': [
                {'type': '.numel()', 'pattern': 'batch_idx.numel() == 0'},
                {'type': '.any()', 'pattern': 'if not has_ko.any()'},
            ],
            '_apply_ko_restrictions': [
                {'type': '.any()', 'pattern': 'if not has_ko.any()'},
            ],
            '_process_captures': [
                {'type': '.any()', 'pattern': 'if not seeds.any()'},
                {'type': '.any()', 'pattern': 'if captured.any()'},
            ],
            '_flood_fill': [
                {'type': '.any()', 'pattern': 'if not expanded.any()'},
            ],
            'legal_moves': [
                {'type': '.any()', 'pattern': 'if not active_games.any()'},
            ],
        }
 
        """Analyze methods for CPU-GPU sync patterns"""
        sync_patterns = {}
        
        # Known sync points in current code
        problem_methods = {
            '_detect_ko': [
                {'type': '.numel()', 'pattern': 'batch_idx.numel() == 0'},
                {'type': '.any()', 'pattern': 'if not has_ko.any()'},
            ],
            '_apply_ko_restrictions': [
                {'type': '.any()', 'pattern': 'if not has_ko.any()'},
            ],
            '_process_captures': [
                {'type': '.any()', 'pattern': 'if not seeds.any()'},
                {'type': '.any()', 'pattern': 'if captured.any()'},
            ],
            '_flood_fill': [
                {'type': '.any()', 'pattern': 'if not expanded.any()'},
            ],
            'legal_moves': [
                {'type': '.any()', 'pattern': 'if not active_games.any()'},
            ],
        }
        
        # Note: .any() is actually optimized by PyTorch, but it's good to track
        return problem_methods