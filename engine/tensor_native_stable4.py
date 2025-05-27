"""tensor_board_standard.py - Refactored Go engine with explicit tensor patterns.

This version emphasizes:
1. Explicit dimension naming and documentation
2. Type hints with tensor shapes
3. Separation of concerns
4. Clear tensor operation patterns
"""

from __future__ import annotations
import os
from typing import Tuple, Optional, NamedTuple, TYPE_CHECKING
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np

#  Fallback settings for both CUDA and MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # Optional: helps with debugging

# ========================= CONSTANTS & TYPES =========================

# Dimension naming convention
# B = Batch size (number of parallel games)
# C = Channels (0=black, 1=white)
# H = Height (board size)
# W = Width (board size)

@dataclass
class Dims:
    """Standard dimension names for clarity"""
    BATCH = 0
    CHANNEL = 1
    HEIGHT = 2
    WIDTH = 3

# Game constants
class Stone:
    BLACK = 0
    WHITE = 1
    EMPTY = -1

# Type aliases for clarity
BatchTensor = Tensor  # Shape: (B, ...)
BoardTensor = Tensor  # Shape: (B, H, W)
StoneTensor = Tensor  # Shape: (B, C, H, W)
PositionTensor = Tensor  # Shape: (B, 2) where 2=[row, col]

# ========================= DEVICE MANAGEMENT =========================

#  Device selection with CUDA > MPS > CPU priority
def select_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ========================= TENSOR OPERATIONS =========================
class TensorOps:
    """Reusable tensor operations for Go logic"""
    
    @staticmethod
    def create_neighbor_kernel(device: torch.device) -> Tensor:
        """Create 3x3 convolution kernel for neighbor counting
        
        Returns:
            Tensor of shape (1, 1, 3, 3) - conv2d compatible
        """
        kernel = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=device)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    @staticmethod
    def count_neighbors(board: BoardTensor, kernel: Tensor) -> BoardTensor:
        """Count orthogonal neighbors using convolution
        
        Args:
            board: Shape (B, H, W) - binary mask
            kernel: Shape (1, 1, 3, 3) - neighbor kernel
            
        Returns:
            Shape (B, H, W) - neighbor counts
        """
        # Add channel dimension for conv2d: (B, H, W) -> (B, 1, H, W)
        board_4d = board.unsqueeze(1).float()
        counts = F.conv2d(board_4d, kernel, padding=1)
        return counts.squeeze(1)  # Remove channel: (B, 1, H, W) -> (B, H, W)

# ========================= MAIN GO ENGINE =========================

class TensorBoard(torch.nn.Module):
    """Fully vectorized Go board implementation
    
    Tensor Shape Convention:
        - Batch operations: (B, ...)
        - Board state: (B, C, H, W) where C=2 for black/white
        - Positions: (B, 2) for [row, col] coordinates
        - Masks: (B, H, W) for board-shaped boolean operations
    """
    
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        # Core parameters
        self.batch_size = batch_size
        self.board_size = board_size
        self.device = device or select_device()
        
        # Initialize tensor operations
        self.ops = TensorOps()
        
        # Register constant buffers
        self._register_constants()
        
        # Register mutable state
        self._register_state()
        
        # Cache for finished games (efficiency optimization)
        self._finished_games_cache = None
        self._cache_valid = False
    
    def _register_constants(self) -> None:
        """Register immutable tensors as buffers"""
        # Neighbor counting kernel: (1, 1, 3, 3)
        self.register_buffer(
            "neighbor_kernel",
            self.ops.create_neighbor_kernel(self.device)
        )
        
        # Zobrist hashing for position tracking
        torch.manual_seed(42)
        max_hash = torch.iinfo(torch.int64).max
        
        # Hash keys for each position and color: (C, H, W)
        self.register_buffer(
            "zobrist_position_keys",
            torch.randint(0, max_hash, (2, self.board_size, self.board_size), 
                         dtype=torch.int64, device=self.device)
        )
        
        # Hash keys for turn: (2,)
        self.register_buffer(
            "zobrist_turn_keys",
            torch.randint(0, max_hash, (2,), dtype=torch.int64, device=self.device)
        )
    
    def _register_state(self) -> None:
        """Register mutable game state as buffers"""
        B, H, W = self.batch_size, self.board_size, self.board_size
        
        # Stone positions: (B, C, H, W) - C=2 for black/white
        self.register_buffer(
            "stones",
            torch.zeros((B, 2, H, W), dtype=torch.bool, device=self.device)
        )
        
        # Current player: (B,) - 0=black, 1=white
        self.register_buffer(
            "current_player",
            torch.zeros(B, dtype=torch.uint8, device=self.device)
        )
        
        # Position hash for ko detection: (B,)
        self.register_buffer(
            "position_hash",
            torch.zeros(B, dtype=torch.int64, device=self.device)
        )
        
        # Ko points: (B, 2) - [row, col], -1 if no ko
        self.register_buffer(
            "ko_points",
            torch.full((B, 2), -1, dtype=torch.int16, device=self.device)
        )
        
        # Pass counter for game end: (B,)
        self.register_buffer(
            "pass_count",
            torch.zeros(B, dtype=torch.uint8, device=self.device)
        )
    
    # ==================== BOARD QUERIES ====================
    
    def get_empty_mask(self) -> BoardTensor:
        """Get mask of empty positions
        
        Returns:
            Shape (B, H, W) - True where empty
        """
        occupied = self.stones[:, Stone.BLACK] | self.stones[:, Stone.WHITE]
        return ~occupied
    
    def get_current_player_stones(self) -> BoardTensor:
        """Get stones of current player
        
        Returns:
            Shape (B, H, W) - True where current player has stones
        """
        batch_idx = torch.arange(self.batch_size, device=self.device)
        return self.stones[batch_idx, self.current_player.long()]
    
    def get_opponent_stones(self) -> BoardTensor:
        """Get stones of opponent
        
        Returns:
            Shape (B, H, W) - True where opponent has stones
        """
        batch_idx = torch.arange(self.batch_size, device=self.device)
        opponent = 1 - self.current_player
        return self.stones[batch_idx, opponent.long()]
    
    # ==================== LEGAL MOVE DETECTION ====================
    
    def legal_moves(self) -> BoardTensor:
        """Compute legal moves for current player
        
        Returns:
            Shape (B, H, W) - True where move is legal
        """
        # Early exit: finished games have no legal moves
        finished = self.is_game_over()
        
        # Initialize all as illegal
        legal = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                           dtype=torch.bool, device=self.device)
        
        # If all games are finished, return empty legal moves
        if finished.all():
            return legal
        
        # Only compute for active games
        active = ~finished
        active_indices = active.nonzero(as_tuple=True)[0]
        
        # Get empty mask for active games only
        empty = self.get_empty_mask()
        
        # Rule 1: Can play on empty points with liberties
        liberty_counts = self.ops.count_neighbors(empty, self.neighbor_kernel)
        legal_active = empty & (liberty_counts > 0)
        
        # Rule 2: Can capture opponent groups
        legal_active |= self._check_capture_moves(empty)
        
        # Rule 3: Cannot play ko
        legal_active = self._apply_ko_rule(legal_active)
        
        # Mask out finished games
        # Vectorized masking - no loop!
        legal = legal_active.clone()
        legal[finished] = False
        
        return legal
    
    def _check_capture_moves(self, empty: BoardTensor) -> BoardTensor:
        """Check which empty points would capture opponent stones
        
        Args:
            empty: Shape (B, H, W) - empty positions
            
        Returns:
            Shape (B, H, W) - positions that would capture
        """
        opponent_stones = self.get_opponent_stones()
        
        # Find opponent groups with exactly 1 liberty
        opponent_liberties = self.ops.count_neighbors(empty, self.neighbor_kernel)
        opponent_liberties = opponent_liberties * opponent_stones
        vulnerable = opponent_stones & (opponent_liberties == 1)
        
        # Find empty points adjacent to vulnerable groups
        adjacent_to_vulnerable = self.ops.count_neighbors(vulnerable, self.neighbor_kernel) > 0
        
        return empty & adjacent_to_vulnerable
    
    def _apply_ko_rule(self, legal: BoardTensor) -> BoardTensor:
        """Remove ko points from legal moves
        
        Args:
            legal: Shape (B, H, W) - current legal moves
            
        Returns:
            Shape (B, H, W) - legal moves without ko
        """
        has_ko = self.ko_points[:, 0] >= 0  # Shape: (B,)
        
        if has_ko.any():
            batch_with_ko = has_ko.nonzero(as_tuple=True)[0]
            ko_rows = self.ko_points[batch_with_ko, 0].long()
            ko_cols = self.ko_points[batch_with_ko, 1].long()
            legal[batch_with_ko, ko_rows, ko_cols] = False
        
        return legal
    
    # ==================== MOVE EXECUTION ====================
    
    def step(self, positions: PositionTensor) -> None:
        """Execute moves for all games in batch
        
        Args:
            positions: Shape (B, 2) - [row, col] coordinates
                      Use row=-1 for pass
        """
        if positions.shape != (self.batch_size, 2):
            raise ValueError(f"Expected positions shape ({self.batch_size}, 2), got {positions.shape}")
        
        # Invalidate cache
        self._cache_valid = False
        
        # Get finished games before move
        finished = self.is_game_over()
        
        # Identify passes vs plays
        is_pass = positions[:, 0] < 0  # Shape: (B,)
        is_play = ~is_pass
        
        # Force finished games to pass
        is_play = is_play & ~finished
        is_pass = is_pass | finished
        
        # Update pass counter
        self._update_pass_count(is_pass)
        
        # Clear ko points for active games only
        if not finished.all():
            self.ko_points[~finished] = -1
        
        # Update hash for turn change (only active games)
        if not finished.all():
            self._update_hash_for_turn(~finished)
        
        # Process actual moves
        if is_play.any():
            self._execute_moves(positions, is_play)
        
        # Switch players (only for active games)
        if not finished.all():
            self._switch_players(~finished)
    
    def _update_pass_count(self, is_pass: BatchTensor) -> None:
        """Update pass counter, reset on non-pass"""
        self.pass_count = torch.where(
            is_pass,
            self.pass_count + 1,
            torch.zeros_like(self.pass_count)
        )
    
    def _update_hash_for_turn(self, active_mask: Optional[BatchTensor] = None) -> None:
        """Update position hash for turn change"""
        if active_mask is None:
            self.position_hash ^= self.zobrist_turn_keys[self.current_player.long()]
        else:
            # Only update for active games
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            self.position_hash[active_indices] ^= self.zobrist_turn_keys[
                self.current_player[active_indices].long()
            ]
    
    def _execute_moves(self, positions: PositionTensor, is_play: BatchTensor) -> None:
        """Place stones and handle captures"""
        # Extract active positions
        active_batch = is_play.nonzero(as_tuple=True)[0]
        rows = positions[is_play, 0].long()
        cols = positions[is_play, 1].long()
        colors = self.current_player[is_play].long()
        
        # Place stones
        self.stones[active_batch, colors, rows, cols] = True
        
        # Update hash
        self._update_hash_for_stones(active_batch, rows, cols, colors)
        
        # Process captures
        self._process_captures(active_batch, rows, cols, colors)
    
    def _update_hash_for_stones(
        self, 
        batch_idx: Tensor, 
        rows: Tensor, 
        cols: Tensor, 
        colors: Tensor
    ) -> None:
        """Update position hash for placed stones"""
        # Compute hash updates
        flat_idx = rows * self.board_size + cols
        hash_updates = torch.zeros_like(self.position_hash)
        
        hash_values= torch.gather(
        self.zobrist_position_keys[colors].view(colors.shape[0], -1),
        1,
        flat_idx.unsqueeze(1)
        ).squeeze(1)
        
        self.position_hash[batch_idx] ^= hash_values
    
    def _switch_players(self, active_mask: Optional[BatchTensor] = None) -> None:
        """Switch current player and update hash"""
        if active_mask is None:
            self.current_player ^= 1
            self.position_hash ^= self.zobrist_turn_keys[self.current_player.long()]
        else:
            # Only switch for active games
            self.current_player[active_mask] ^= 1
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            self.position_hash[active_indices] ^= self.zobrist_turn_keys[
                self.current_player[active_indices].long()
            ]
    
    # ==================== CAPTURE LOGIC ====================
    
    def _process_captures(
        self,
        batch_idx: Tensor,
        rows: Tensor,
        cols: Tensor,
        colors: Tensor
    ) -> None:
        """Remove captured opponent groups and track ko"""
        if batch_idx.numel() == 0:
            return
        
        # Find groups to check
        neighbor_groups = self._get_neighbor_opponent_groups(
            batch_idx, rows, cols, colors
        )
        
        if not neighbor_groups.any():
            return
        
        # Flood fill to find complete groups
        complete_groups = self._flood_fill_groups(neighbor_groups)
        
        # Check which groups have no liberties
        captured = self._find_captured_groups(complete_groups)
        
        if not captured.any():
            return
        
        # Remove captured stones and update ko
        self._remove_captured_stones(captured)
    
    def _get_neighbor_opponent_groups(
        self,
        batch_idx: Tensor,
        rows: Tensor,
        cols: Tensor,
        colors: Tensor
    ) -> BoardTensor:
        """Find opponent stones adjacent to played moves"""
        # Implementation details...
        # Returns: Shape (B, H, W)
        B, H, W = self.batch_size, self.board_size, self.board_size
        
        # Create a mask of where moves were played
        move_mask = torch.zeros((B, H, W), dtype=torch.float32, device=self.device)
        move_mask[batch_idx, rows, cols] = 1.0
        
        # Use convolution to find neighbors (reuse existing kernel!)
        neighbors = self.ops.count_neighbors(move_mask, self.neighbor_kernel) > 0
        
        # Filter for opponent stones
        opponent_stones = self.get_opponent_stones()
        return neighbors & opponent_stones
    
    def _flood_fill_groups(self, seeds: BoardTensor) -> BoardTensor:
        """Expand seeds to complete groups using flood fill"""
        opponent_stones = self.get_opponent_stones()
        groups = seeds.clone()
        
        while True:
            # Dilate groups
            expanded = self.ops.count_neighbors(groups, self.neighbor_kernel) > 0
            new_stones = expanded & opponent_stones & ~groups
            
            if not new_stones.any():
                break
                
            groups |= new_stones
        
        return groups
    
    def _find_captured_groups(self, groups: BoardTensor) -> BoardTensor:
        """Identify groups with no liberties"""
        empty = self.get_empty_mask()
        
        # Count liberties for each position in groups
        group_liberties = self.ops.count_neighbors(empty, self.neighbor_kernel)
        group_liberties = group_liberties * groups
        
        # Group has no liberties if no position has liberties
        # This is simplified - proper implementation would track connected components
        return groups & (group_liberties == 0)
    
    def _remove_captured_stones(self, captured: BoardTensor) -> None:
        """Remove captured stones and handle ko"""
        if not captured.any():
            return
        
        # Find captured positions
        captured_positions = captured.nonzero(as_tuple=False)
        batch_idx = captured_positions[:, 0]
        rows = captured_positions[:, 1]
        cols = captured_positions[:, 2]
        
        # Determine stone colors
        opponent_color = (1 - self.current_player[batch_idx]).long()
        
        # Remove stones
        self.stones[batch_idx, opponent_color, rows, cols] = False
        
        # Update hash

        # VECTORIZED hash update - NO LOOP!
        flat_idx = rows * self.board_size + cols
        zobrist_flat = self.zobrist_position_keys.view(2, -1)
        hash_values = zobrist_flat[opponent_color, flat_idx]
        self.position_hash[batch_idx] ^= hash_values

        
        # Check for ko (single stone capture)
        for b in batch_idx.unique():
            batch_captures = (batch_idx == b).sum()
            if batch_captures == 1:
                idx = (batch_idx == b).nonzero(as_tuple=True)[0][0]
                # Direct assignment - more efficient
                self.ko_points[b, 0] = rows[idx]
                self.ko_points[b, 1] = cols[idx]
    

    
    def is_game_over(self) -> BatchTensor:
        """Check if games are finished
        
        Returns:
            Shape (B,) - True where game is over
        """
        # Use cache if valid
        if self._cache_valid and self._finished_games_cache is not None:
            return self._finished_games_cache
        
        # Compute and cache
        self._finished_games_cache = self.pass_count >= 2
        self._cache_valid = True
        return self._finished_games_cache
    
    def compute_scores(self) -> Tensor:
        """Compute current scores
        
        Returns:
            Shape (B, 2) - [black_score, white_score] for each game
        """
        black_stones = self.stones[:, Stone.BLACK].sum(dim=(1, 2)).float()
        white_stones = self.stones[:, Stone.WHITE].sum(dim=(1, 2)).float()
        return torch.stack([black_stones, white_stones], dim=1)
    
    # ==================== FEATURES FOR NN ====================
    
    def extract_features(self) -> Tensor:
        """Extract features for neural network
        
        Returns:
            Shape (B, 5, H, W) - feature planes:
                0: Current player stones
                1: Opponent stones
                2: Legal moves
                3: Liberties of current player groups
                4: Turn indicator (0 or 1 everywhere)
        """
        B, H, W = self.batch_size, self.board_size, self.board_size
        
        # Basic stone features
        current_stones = self.get_current_player_stones().float()
        opponent_stones = self.get_opponent_stones().float()
        
        # Legal moves
        legal = self.legal_moves().float()
        
        # Liberty counts for current player
        empty = self.get_empty_mask().float()
        current_liberties = self.ops.count_neighbors(empty, self.neighbor_kernel)
        current_liberties *= current_stones
        
        # Turn indicator plane
        turn_plane = self.current_player.view(B, 1, 1).expand(B, H, W).float()
        
        return torch.stack([
            current_stones,
            opponent_stones,
            legal,
            current_liberties,
            turn_plane
        ], dim=1)
    
    # ==================== UTILITIES ====================
    
    def to_numpy(self, batch_idx: int = 0) -> "np.ndarray":
        """Convert single board to numpy for visualization
        
        Returns:
            Shape (H, W) - -1=empty, 0=black, 1=white
        """
        import numpy as np
        
        board = np.full((self.board_size, self.board_size), Stone.EMPTY, dtype=np.int8)
        
        black_mask = self.stones[batch_idx, Stone.BLACK].cpu().numpy()
        white_mask = self.stones[batch_idx, Stone.WHITE].cpu().numpy()
        
        board[black_mask] = Stone.BLACK
        board[white_mask] = Stone.WHITE
        
        return board


# ========================= COMPILATION =========================

# Optional: Enable torch.compile for optimization
try:
    TensorBoard = torch.compile(TensorBoard, dynamic=True, fullgraph=False)
except Exception:
    pass  # Compilation not available