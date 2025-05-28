"""tensor_board_optimized.py - Complete optimized Go engine with all performance improvements

Key optimizations included:
1. Direct slicing for neighbor counting (no convolution)
2. Bounded flood fill with pre-allocated buffers
3. Selective cache invalidation
4. Reduced memory allocations
5. In-place operations where possible
6. Optimized tensor indexing
7. Better compilation settings
"""

from __future__ import annotations
import os
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn.functional as F
from torch import Tensor

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

# ========================= OPTIMIZED GO ENGINE =========================

class TensorBoard(torch.nn.Module):
    """Performance-optimized vectorized Go board implementation"""
    
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.board_size = board_size
        self.device = device or self.select_device()
        
        self._init_constants()
        self._init_state()
        self._init_work_buffers()
        self._cache = {}
        self._cache_valid = {}
    
    @staticmethod
    def select_device() -> torch.device:
        """Select best available device: CUDA > MPS > CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _init_constants(self) -> None:
        """Initialize constant tensors"""
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
        
        # Pre-compute maximum flood fill iterations
        self.max_flood_iterations = min(self.board_size * self.board_size // 4, 100)
        
        # Pre-allocate batch index range for repeated use
        self.register_buffer(
            "_batch_range",
            torch.arange(self.batch_size, device=self.device, dtype=torch.long)
        )
    
    def _init_state(self) -> None:
        """Initialize mutable game state"""
        B, H, W = self.batch_size, self.board_size, self.board_size
        
        self.register_buffer("stones", torch.zeros((B, 2, H, W), dtype=torch.bool, device=self.device))
        self.register_buffer("current_player", torch.zeros(B, dtype=torch.uint8, device=self.device))
        self.register_buffer("position_hash", torch.zeros(B, dtype=torch.int64, device=self.device))
        self.register_buffer("ko_points", torch.full((B, 2), -1, dtype=torch.int16, device=self.device))
        self.register_buffer("pass_count", torch.zeros(B, dtype=torch.uint8, device=self.device))
    
    def _init_work_buffers(self) -> None:
        """Pre-allocate work buffers for operations"""
        B, H, W = self.batch_size, self.board_size, self.board_size
        
        # Work buffers for flood fill
        self.register_buffer("_flood_work", torch.zeros((B, H, W), dtype=torch.bool, device=self.device))
        self.register_buffer("_flood_expanded", torch.zeros((B, H, W), dtype=torch.bool, device=self.device))
        
        # Work buffer for neighbor counting
        self.register_buffer("_neighbor_work", torch.zeros((B, H, W), dtype=torch.float32, device=self.device))
        
        # Work buffer for move masks
        self.register_buffer("_move_mask", torch.zeros((B, H, W), dtype=torch.bool, device=self.device))
    
    # ==================== OPTIMIZED UTILITIES ====================
    
    def _count_neighbors(self, mask: BoardTensor) -> BoardTensor:
        """Count orthogonal neighbors - optimized without convolution"""
        # Clear work buffer
        self._neighbor_work.zero_()
        
        # Add neighbors using slicing (much faster than convolution)
        self._neighbor_work[:, 1:, :] += mask[:, :-1, :].float()    # from top
        self._neighbor_work[:, :-1, :] += mask[:, 1:, :].float()    # from bottom
        self._neighbor_work[:, :, 1:] += mask[:, :, :-1].float()    # from left
        self._neighbor_work[:, :, :-1] += mask[:, :, 1:].float()    # from right
        
        return self._neighbor_work
    
    def _update_hash(self, positions: Tuple[Tensor, Tensor, Tensor], colors: Tensor, add: bool = True):
        """Update position hash for stone changes"""
        batch_idx, rows, cols = positions
        if batch_idx.numel() == 0:
            return
            
        flat_idx = rows * self.board_size + cols
        hash_values = self.zobrist_stones[colors].view(colors.shape[0], -1)
        hash_values = hash_values.gather(1, flat_idx.unsqueeze(1)).squeeze(1)
        
        # In-place XOR
        self.position_hash[batch_idx] ^= hash_values
    
    @with_active_games
    def _toggle_turn(self, active_mask: BatchTensor) -> None:
        """Switch current player and update hash"""
        # Update hash for old player
        self.position_hash[active_mask] ^= self.zobrist_turn[self.current_player[active_mask].long()]
        # Switch player in-place
        self.current_player[active_mask] ^= 1
        # Update hash for new player
        self.position_hash[active_mask] ^= self.zobrist_turn[self.current_player[active_mask].long()]
    
    def _invalidate_cache(self, full: bool = False) -> None:
        """Invalidate cache - selective by default"""
        if full:
            self._cache.clear()
            self._cache_valid.clear()
        else:
            # Only invalidate move-dependent caches
            self._cache_valid['empty'] = False
            self._cache_valid['legal'] = False
    
    # ==================== BOARD QUERIES ====================
    
    @property
    def empty_mask(self) -> BoardTensor:
        """Get empty positions (cached)"""
        if not self._cache_valid.get('empty', False):
            # Use any() which is faster than creating intermediate tensor
            self._cache['empty'] = ~self.stones.any(dim=1)
            self._cache_valid['empty'] = True
        return self._cache['empty']
    
    def get_player_stones(self, player: Optional[int] = None) -> BoardTensor:
        """Get stones for specified player (None = current)"""
        if player is None:
            # Use pre-allocated batch range
            return self.stones[self._batch_range, self.current_player.long()]
        return self.stones[:, player]
    
    def get_opponent_stones(self) -> BoardTensor:
        """Get opponent's stones"""
        opponent = 1 - self.current_player
        return self.stones[self._batch_range, opponent.long()]
    
    # ==================== KO HANDLING ====================
    
    def _apply_ko_restrictions(self, legal: BoardTensor) -> BoardTensor:
        """Apply ko restrictions to legal moves"""
        has_ko = self.ko_points[:, 0] >= 0
        if not has_ko.any():
            return legal
        
        # Direct indexing for ko points
        ko_batch = has_ko.nonzero(as_tuple=True)[0]
        if ko_batch.numel() == 0:
            return legal
            
        ko_rows = self.ko_points[ko_batch, 0].long()
        ko_cols = self.ko_points[ko_batch, 1].long()
        
        # Clone only if necessary
        if ko_batch.numel() > 0:
            legal = legal.clone()
            legal[ko_batch, ko_rows, ko_cols] = False
        
        return legal
    
    def _detect_ko(self, captured_positions: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Detect and set ko points for single stone captures"""
        batch_idx, rows, cols = captured_positions
        
        if batch_idx.numel() == 0:
            return
        
        # Count captures per batch efficiently
        capture_counts = torch.bincount(batch_idx, minlength=self.batch_size)
        
        # Find batches with exactly one capture
        single_capture_batches = (capture_counts == 1)
        
        if not single_capture_batches.any():
            return
        
        # Build index mapping
        batch_to_idx = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        idx_range = torch.arange(len(batch_idx), device=self.device)
        batch_to_idx.scatter_(0, batch_idx, idx_range)
        
        # Extract positions for single capture batches
        single_capture_idx = batch_to_idx[single_capture_batches]
        valid_idx = single_capture_idx >= 0
        
        if valid_idx.any():
            valid_batches = single_capture_batches.nonzero(as_tuple=True)[0][valid_idx]
            valid_indices = single_capture_idx[valid_idx]
            
            self.ko_points[valid_batches, 0] = rows[valid_indices].to(torch.int16)
            self.ko_points[valid_batches, 1] = cols[valid_indices].to(torch.int16)
    
    # ==================== OPTIMIZED FLOOD FILL ====================
    
    def _flood_fill(self, seeds: BoardTensor, mask: BoardTensor) -> BoardTensor:
        """Optimized flood fill with bounded iterations and pre-allocated buffers"""
        if not seeds.any():
            return seeds
            
        groups = seeds.clone()
        
        # Use pre-allocated work buffers
        for _ in range(self.max_flood_iterations):
            # Clear expanded buffer
            self._flood_expanded.zero_()
            
            # Get neighbors using slicing (avoiding convolution)
            # Top neighbors
            self._flood_expanded[:, 1:, :] |= groups[:, :-1, :]
            # Bottom neighbors  
            self._flood_expanded[:, :-1, :] |= groups[:, 1:, :]
            # Left neighbors
            self._flood_expanded[:, :, 1:] |= groups[:, :, :-1]
            # Right neighbors
            self._flood_expanded[:, :, :-1] |= groups[:, :, 1:]
            
            # Mask to valid positions and exclude already visited
            self._flood_expanded &= mask
            self._flood_expanded &= ~groups
            
            # Check if we found new positions
            if not self._flood_expanded.any():
                break
            
            # Add to groups in-place
            groups |= self._flood_expanded
        
        return groups
    
    # ==================== OPTIMIZED LEGAL MOVES ====================
    
    def legal_moves(self) -> BoardTensor:
        """Compute legal moves for current player"""
        if self._cache_valid.get('legal', False):
            return self._cache['legal']
        
        # No moves if game over
        finished = self.is_game_over()
        if finished.all():
            legal = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                              dtype=torch.bool, device=self.device)
            self._cache['legal'] = legal
            self._cache_valid['legal'] = True
            return legal
        
        # Get empty positions
        empty = self.empty_mask
        
        # Count empty neighbors (positions with liberties)
        empty_neighbors = self._count_neighbors(empty) > 0
        
        # Start with empty positions that have liberties
        legal = empty & empty_neighbors
        
        # Add capture moves
        capture_moves = self._find_capture_moves()
        if capture_moves is not None:
            legal |= capture_moves
        
        # Apply ko restrictions
        legal = self._apply_ko_restrictions(legal)
        
        # Mask finished games
        if finished.any():
            legal &= ~finished.view(-1, 1, 1)
        
        self._cache['legal'] = legal
        self._cache_valid['legal'] = True
        return legal
    
    def _find_capture_moves(self) -> Optional[BoardTensor]:
        """Find moves that would capture opponent stones - optimized"""
        opponent = self.get_opponent_stones()
        if not opponent.any():
            return None
            
        empty = self.empty_mask
        
        # Count liberties for opponent stones using cached neighbor count
        liberties = self._count_neighbors(empty)
        
        # Find opponent stones with exactly 1 liberty
        vulnerable = opponent & (liberties == 1)
        
        if not vulnerable.any():
            return None
        
        # Find empty points adjacent to vulnerable stones
        vulnerable_neighbors = self._count_neighbors(vulnerable) > 0
        
        return empty & vulnerable_neighbors
    
    # ==================== MOVE EXECUTION ====================
    
    def step(self, positions: PositionTensor) -> None:
        """Execute moves for all games"""
        self._invalidate_cache()
        
        # Classify moves
        is_pass = positions[:, 0] < 0
        finished = self.is_game_over()
        is_play = ~is_pass & ~finished
        
        # Update pass counter efficiently
        if is_pass.any() or finished.any():
            self.pass_count = torch.where(
                is_pass | finished,
                torch.where(is_pass, self.pass_count + 1, self.pass_count),
                torch.zeros_like(self.pass_count)
            )
        else:
            self.pass_count.zero_()
        
        # Reset ko for non-finished games
        if not finished.all():
            active = ~finished
            self.ko_points[active] = -1
        
        # Execute moves
        if is_play.any():
            play_idx = is_play.nonzero(as_tuple=True)[0]
            self._place_stones(positions[is_play], play_idx)
        
        # Switch turn
        self._toggle_turn()
    
    def _place_stones(self, positions: PositionTensor, batch_idx: Tensor) -> None:
        """Place stones and handle captures"""
        if batch_idx.numel() == 0:
            return
            
        rows, cols = positions[:, 0].long(), positions[:, 1].long()
        colors = self.current_player[batch_idx].long()
        
        # Place stones
        self.stones[batch_idx, colors, rows, cols] = True
        self._update_hash((batch_idx, rows, cols), colors, add=True)
        
        # Handle captures
        self._process_captures(batch_idx, rows, cols)
    
    def _process_captures(self, batch_idx: Tensor, rows: Tensor, cols: Tensor) -> None:
        """Remove captured opponent groups - optimized"""
        if batch_idx.numel() == 0:
            return
            
        # Clear and create move mask
        self._move_mask.zero_()
        self._move_mask[batch_idx, rows, cols] = True
        
        # Find adjacent positions
        neighbors = self._count_neighbors(self._move_mask) > 0
        
        # Get opponent stones
        opponent_stones = self.get_opponent_stones()
        
        # Find adjacent opponent stones
        seeds = neighbors & opponent_stones
        
        if not seeds.any():
            return
        
        # Flood fill to find complete groups
        groups = self._flood_fill(seeds, opponent_stones)
        
        if not groups.any():
            return
        
        # Find captured groups (no liberties)
        # Reuse empty mask calculation
        empty = self.empty_mask
        group_liberties = self._count_neighbors(empty)
        captured = groups & (group_liberties == 0)
        
        if captured.any():
            self._remove_captured_stones(captured)
    
    def _remove_captured_stones(self, captured: BoardTensor) -> None:
        """Remove captured stones and handle ko"""
        positions = captured.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return
            
        batch_idx = positions[:, 0]
        rows = positions[:, 1]
        cols = positions[:, 2]
        
        # Determine colors and remove stones
        colors = (1 - self.current_player[batch_idx]).long()
        self.stones[batch_idx, colors, rows, cols] = False
        
        # Update hash
        self._update_hash((batch_idx, rows, cols), colors, add=False)
        
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
    
    def extract_features(self) -> Tensor:
        """Extract features for neural network - optimized"""
        current = self.get_player_stones().float()
        opponent = self.get_opponent_stones().float()
        legal = self.legal_moves().float()
        
        # Reuse neighbor counting for liberties
        empty = self.empty_mask
        current_liberties = self._count_neighbors(empty) * current
        
        # Create turn feature
        turn = self.current_player.float().view(-1, 1, 1).expand(-1, self.board_size, self.board_size)
        
        return torch.stack([current, opponent, legal, current_liberties, turn], dim=1)
    
    def to_numpy(self, batch_idx: int = 0):
        """Convert to numpy for visualization"""
        import numpy as np
        board = np.full((self.board_size, self.board_size), Stone.EMPTY, dtype=np.int8)
        
        # Get stone positions
        black_positions = self.stones[batch_idx, Stone.BLACK].cpu().numpy()
        white_positions = self.stones[batch_idx, Stone.WHITE].cpu().numpy()
        
        board[black_positions] = Stone.BLACK
        board[white_positions] = Stone.WHITE
        
        return board


# ==================== COMPILATION SETTINGS ====================

def compile_optimized_board():
    """Compile the board with optimal settings"""
    try:
        compiled = torch.compile(
            TensorBoard,
            mode="max-autotune",      # Maximum optimization
            dynamic=True,             # Handle dynamic batch sizes
            fullgraph=False,          # Allow Python control flow
            backend="inductor",       # Use inductor backend
        )
        return compiled
    except Exception as e:
        print(f"Compilation failed: {e}")
        return TensorBoard


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Create optimized board
    board = TensorBoard(batch_size=512, board_size=19)
    
    # Or with compilation
    BoardClass = compile_optimized_board()
    board = BoardClass(batch_size=512, board_size=19)
    
    # Example game simulation
    import time
    
    start_time = time.time()
    moves_played = 0
    
    while not board.is_game_over().all() and moves_played < 1000:
        # Get legal moves
        legal = board.legal_moves()
        
        # Sample random moves
        positions = torch.full((board.batch_size, 2), -1, dtype=torch.int32, device=board.device)
        
        for b in range(board.batch_size):
            if board.is_game_over()[b]:
                continue
                
            legal_positions = legal[b].nonzero(as_tuple=False)
            if legal_positions.numel() > 0:
                # Random selection
                idx = torch.randint(0, legal_positions.shape[0], (1,)).item()
                positions[b] = legal_positions[idx]
        
        # Execute moves
        board.step(positions)
        moves_played += board.batch_size
    
    elapsed = time.time() - start_time
    print(f"Played {moves_played} moves in {elapsed:.2f}s")
    print(f"Speed: {moves_played/elapsed:.0f} moves/second")
    
    # Get final scores
    scores = board.compute_scores()
    print(f"\nFinal scores:")
    print(f"Black average: {scores[:, 0].mean():.1f}")
    print(f"White average: {scores[:, 1].mean():.1f}")