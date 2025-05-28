"""tensor_native.py - Go engine.

Key improvements:
1. Uses centralized utilities from utils.shared
2. Cleaner separation of concerns
3. More functional programming style
"""

from __future__ import annotations
import os
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from functools import wraps

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
    """Elegant vectorized Go board implementation"""
    
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.board_size = board_size
        self.device = device or select_device()
        
        self._init_constants()
        self._init_state()
        self._cache = {}
    
    def _init_constants(self) -> None:
        """Initialize constant tensors"""
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
        B, H, W = self.batch_size, self.board_size, self.board_size
        
        self.register_buffer("stones", torch.zeros((B, 2, H, W), dtype=torch.bool, device=self.device))
        self.register_buffer("current_player", torch.zeros(B, dtype=torch.uint8, device=self.device))
        self.register_buffer("position_hash", torch.zeros(B, dtype=torch.int64, device=self.device))
        self.register_buffer("ko_points", create_pass_positions(B, self.device))
        self.register_buffer("pass_count", torch.zeros(B, dtype=torch.uint8, device=self.device))
    
    # ==================== CORE UTILITIES ====================
    
    def _count_neighbors(self, mask: BoardTensor) -> BoardTensor:
        """Count orthogonal neighbors"""
        mask_4d = mask.unsqueeze(1).float()
        counts = F.conv2d(mask_4d, self.neighbor_kernel, padding=1)
        return counts.squeeze(1)
    
    def _update_hash_for_positions(self, positions: Tuple[Tensor, Tensor, Tensor], colors: Tensor):
        """Update position hash for stone changes (unified method)"""
        batch_idx, rows, cols = positions
        flat_idx = coords_to_flat(rows, cols, self.board_size)
        hash_values = self.zobrist_stones[colors].view(colors.shape[0], -1)
        hash_values = hash_values.gather(1, flat_idx.unsqueeze(1)).squeeze(1)
        
        # XOR is its own inverse, so same operation for add/remove
        self.position_hash[batch_idx] ^= hash_values
    
    @with_active_games
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
            occupied = self.stones.any(dim=1)
            self._cache['empty'] = ~occupied
        return self._cache['empty']
    
    def get_player_stones(self, player: Optional[int] = None) -> BoardTensor:
        """Get stones for specified player (None = current)"""
        if player is None:
            batch_idx = get_batch_indices(self.batch_size, self.device)
            return self.stones[batch_idx, self.current_player.long()]
        return self.stones[:, player]
    
    def get_opponent_stones(self) -> BoardTensor:
        """Get opponent's stones"""
        batch_idx = get_batch_indices(self.batch_size, self.device)
        opponent = 1 - self.current_player
        return self.stones[batch_idx, opponent.long()]
    
    def get_playable_games(self, legal_moves: Optional[BoardTensor] = None) -> BatchTensor:
        """Get mask of games that have at least one legal move.
        
        Args:
            legal_moves: Optional pre-computed legal moves tensor.
                        If None, will compute legal moves.
        
        Returns:
            Boolean mask of shape (B,) - True where game has legal moves
        """
        if legal_moves is None:
            legal_moves = self.legal_moves()
        
        return find_playable_games(legal_moves)
    
    # ==================== KO HANDLING ====================
    
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
    
    def _detect_ko(self, captured_positions: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Detect and set ko points for single stone captures (fully vectorized, no loops)"""
        batch_idx, rows, cols = captured_positions
        
        if batch_idx.numel() == 0:
            return
        
        # Count captures per batch using bincount (most efficient for this)
        capture_counts = torch.bincount(batch_idx, minlength=self.batch_size)
        
        # Find batches with exactly one capture
        single_capture_batches = (capture_counts == 1)
        
        if not single_capture_batches.any():
            return
        
        # For batches with single captures, we need to find which index corresponds to each batch
        # scatter_ will keep the last value written, but for single captures there's only one
        batch_to_idx = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        idx_range = torch.arange(len(batch_idx), device=self.device)
        batch_to_idx.scatter_(0, batch_idx, idx_range)
        
        # Extract positions for single capture batches
        single_capture_idx = batch_to_idx[single_capture_batches]
        
        # Update ko points only for batches with single captures
        self.ko_points[single_capture_batches, 0] = rows[single_capture_idx].to(torch.int16)
        self.ko_points[single_capture_batches, 1] = cols[single_capture_idx].to(torch.int16)
    
    # ==================== LEGAL MOVES ====================
    
    def legal_moves(self) -> BoardTensor:
        """Compute legal moves for current player"""
        if 'legal' in self._cache:
            return self._cache['legal']
        
        # Get active (non-finished) games
        active_games = ~self.is_game_over()
        
        # No moves if all games are over
        if not active_games.any():
            legal = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                              dtype=torch.bool, device=self.device)
            self._cache['legal'] = legal
            return legal
        
        # Start with empty positions that have liberties
        legal = self.empty_mask & (self._count_neighbors(self.empty_mask) > 0)
        
        # Add capture moves
        legal |= self._find_capture_moves()
        
        # Apply ko restrictions
        legal = self._apply_ko_restrictions(legal)
        
        # Mask finished games (no legal moves for finished games)
        legal[~active_games] = False
        
        self._cache['legal'] = legal
        return legal
    
    def _find_capture_moves(self) -> BoardTensor:
        """Find moves that would capture opponent stones"""
        opponent = self.get_opponent_stones()
        
        # Find opponent stones with exactly 1 liberty
        liberties = self._count_neighbors(self.empty_mask) * opponent
        vulnerable = opponent & (liberties == 1)
        
        # Find empty points adjacent to vulnerable stones
        return self.empty_mask & (self._count_neighbors(vulnerable) > 0)
    
    # ==================== MOVE EXECUTION ====================
    
    def step(self, positions: PositionTensor) -> None:
        """Execute moves for all games"""
        self._invalidate_cache()
        
        # Classify moves using shared utility
        is_pass = is_pass_move(positions)
        finished = self.is_game_over()
        is_play = ~is_pass & ~finished
        
        # Update pass counter
        self.pass_count = torch.where(
            is_pass | finished,
            torch.where(is_pass, self.pass_count + 1, self.pass_count),
            torch.zeros_like(self.pass_count)
        )
        
        # Reset ko for non-finished games
        if not finished.all():
            self.ko_points[~finished] = -1
        
        # Execute moves
        if is_play.any():
            self._place_stones(positions[is_play], is_play.nonzero(as_tuple=True)[0])
        
        # Switch turn
        self._toggle_turn()
    
    def _place_stones(self, positions: PositionTensor, batch_idx: Tensor) -> None:
        """Place stones and handle captures"""
        rows, cols = positions[:, 0].long(), positions[:, 1].long()
        colors = self.current_player[batch_idx].long()
        
        # Place stones
        self.stones[batch_idx, colors, rows, cols] = True
        self._update_hash_for_positions((batch_idx, rows, cols), colors)
        
        # Handle captures
        self._process_captures(batch_idx, rows, cols)
    
    def _process_captures(self, batch_idx: Tensor, rows: Tensor, cols: Tensor) -> None:
        """Remove captured opponent groups"""
        # Find adjacent opponent stones
        move_mask = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                               dtype=torch.float32, device=self.device)
        move_mask[batch_idx, rows, cols] = 1.0
        
        neighbors = self._count_neighbors(move_mask) > 0
        opponent_stones = self.get_opponent_stones()
        seeds = neighbors & opponent_stones
        
        if not seeds.any():
            return
        
        # Flood fill to find complete groups
        groups = self._flood_fill(seeds, opponent_stones)
        
        # Find captured groups (no liberties)
        group_liberties = self._count_neighbors(self.empty_mask) * groups
        captured = groups & (group_liberties == 0)
        
        if captured.any():
            self._remove_captured_stones(captured)
    
    def _flood_fill(self, seeds: BoardTensor, mask: BoardTensor) -> BoardTensor:
        """Expand seeds to complete groups"""
        groups = seeds.clone()
        
        while True:
            expanded = (self._count_neighbors(groups) > 0) & mask & ~groups
            if not expanded.any():
                break
            groups |= expanded
        
        return groups
    
    def _remove_captured_stones(self, captured: BoardTensor) -> None:
        """Remove captured stones and handle ko"""
        positions = captured.nonzero(as_tuple=False)
        batch_idx = positions[:, 0]
        rows = positions[:, 1]
        cols = positions[:, 2]
        
        # Determine colors and remove stones
        colors = (1 - self.current_player[batch_idx]).long()
        self.stones[batch_idx, colors, rows, cols] = False
        
        # Update hash using unified method
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
    
    def extract_features(self) -> Tensor:
        """Extract features for neural network"""
        current = self.get_player_stones().float()
        opponent = self.get_opponent_stones().float()
        legal = self.legal_moves().float()
        liberties = self._count_neighbors(self.empty_mask) * current
        turn = self.current_player.view(-1, 1, 1).expand(-1, self.board_size, self.board_size).float()
        
        return torch.stack([current, opponent, legal, liberties, turn], dim=1)
    
    def to_numpy(self, batch_idx: int = 0):
        """Convert to numpy for visualization"""
        import numpy as np
        board = np.full((self.board_size, self.board_size), Stone.EMPTY, dtype=np.int8)
        board[self.stones[batch_idx, Stone.BLACK].cpu().numpy()] = Stone.BLACK
        board[self.stones[batch_idx, Stone.WHITE].cpu().numpy()] = Stone.WHITE
        return board