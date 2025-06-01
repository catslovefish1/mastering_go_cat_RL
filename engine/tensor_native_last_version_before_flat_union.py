"""tensor_native.py - Optimized Go engine.

Key improvements:
1. Uses centralized utilities from utils.shared
2. Cleaner separation of concerns
3. More functional programming style
4. Optimized hash updates with unified Zobrist table
5. Improved ko detection without CPU-GPU sync
6. FIXED: Proper capture detection for multiple groups
7. FIXED: Suicide prevention
8. FIXED: Single active game computation per step
9. FIXED: Configurable history depth with history_factor
"""

from __future__ import annotations
import os
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor

# Import shared utilities
from utils.shared import (
    select_device,
    is_pass_move,
    find_playable_games,
    get_batch_indices,
)

# Environment setup
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ========================= CONSTANTS =========================

@dataclass(frozen=True)
class Stone:
    """Stone color constants"""
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1

# Type aliases
BoardTensor = Tensor  # (B, H, W)
StoneTensor = Tensor  # (B, C, H, W)
PositionTensor = Tensor  # (B, 2)
BatchTensor = Tensor  # (B,)

# ========================= GO ENGINE =========================

class TensorBoard(torch.nn.Module):
    """Elegant vectorized Go board implementation"""
    
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        history_factor: int = 10,  # Controls history depth (board_size² × history_factor)
        device: Optional[torch.device] = None,
        enable_timing: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.board_size = board_size
        self.history_factor = history_factor
        self.device = device or select_device()
        self.enable_timing = enable_timing
        
        self._init_zobrist_table()
        self._init_constants()
        self._init_state()
        self._cache = {}
    
    def _init_constants(self) -> None:
        """Initialize constant tensors"""
        # Neighbor kernel for convolutions
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
                            dtype=torch.float32, device=self.device)
        self.register_buffer("neighbor_kernel", kernel.unsqueeze(0).unsqueeze(0))
    
    def _init_zobrist_table(self) -> None:
        """Initialize unified Zobrist hash table with configurable history depth"""
        torch.manual_seed(42)
        max_hash = torch.iinfo(torch.int64).max
        
        # Create Zobrist table for current position (2 colors × board positions)
        stones_size = self.history_factor * self.board_size * self.board_size
        
        # Total size includes stones + turn states
        total_size = stones_size + 2
        
        self.register_buffer(
            "zobrist_unified", 
            torch.randint(0, max_hash, (total_size,), dtype=torch.int64, device=self.device)
        )
        
        # View for stone positions (2 colors × board_size²)
        self.register_buffer(
            "zobrist_stones_flat",
            self.zobrist_unified[:stones_size].view(self.history_factor, self.board_size * self.board_size)
        )
        
        # View for turn indicator (last 2 elements)
        self.register_buffer(
            "zobrist_turn",
            self.zobrist_unified[stones_size:stones_size + 2]
        )
    
    def _init_state(self) -> None:
        """Initialize mutable game state"""
        B, H, W = self.batch_size, self.board_size, self.board_size
        
        self.register_buffer("stones", torch.zeros((B, 2, H, W), dtype=torch.bool, device=self.device))
        self.register_buffer("current_player", torch.zeros(B, dtype=torch.uint8, device=self.device))
        self.register_buffer("position_hash", torch.zeros(B, dtype=torch.int64, device=self.device))
        self.register_buffer("ko_points", torch.full((B, 2), -1, dtype=torch.int8, device=self.device))
        self.register_buffer("pass_count", torch.zeros(B, dtype=torch.uint8, device=self.device))
        self._init_union_find_table()
        
        # Board history with configurable depth using history_factor
        max_moves = self.board_size * self.board_size * self.history_factor
        self.register_buffer(
            "board_history", 
            torch.full((B, max_moves, H * W), -1, dtype=torch.int8, device=self.device)
        )
        self.register_buffer("move_count", torch.zeros(B, dtype=torch.int32, device=self.device))
    
    def _init_union_find_table(self) -> None:
        """Initialize flattened union-find table for efficient group tracking"""
        B, H, W = self.batch_size, self.board_size, self.board_size
        N_squared = H * W
        
        self.register_buffer(
            "flatten_union_find", 
            torch.zeros((B, N_squared, 3), dtype=torch.int32, device=self.device)
        )
        
        # Initialize color to empty
        self.flatten_union_find[:, :, 0] = -1
        
        # Initialize parent indices to self
        indices = torch.arange(N_squared, device=self.device)
        self.flatten_union_find[:, :, 1] = indices.unsqueeze(0).expand(B, -1)

    # ==================== CORE UTILITIES ====================
    
    def _count_neighbors(self, mask: BoardTensor) -> BoardTensor:
        """Count orthogonal neighbors"""
        mask_4d = mask.unsqueeze(1).float()
        counts = F.conv2d(mask_4d, self.neighbor_kernel, padding=1)
        result = counts.squeeze(1)
        return result
    
    def _update_hash_for_positions(self, positions: Tuple[Tensor, Tensor, Tensor], colors: Tensor):
        """Update position hash for stone changes"""
        batch_idx, rows, cols = positions
        
        flat_idx = rows * self.board_size + cols
        hash_values = self.zobrist_stones_flat[colors, flat_idx]
        self.position_hash[batch_idx] ^= hash_values
    
    def _toggle_turn_indexed(self, game_indices: Tensor) -> None:
        """Switch current player and update hash for specified games"""
        if game_indices.numel() == 0:
            return
            
        # Update hash for old player
        old_players = self.current_player[game_indices].long()
        self.position_hash[game_indices] ^= self.zobrist_turn[old_players]
        
        # Switch player
        self.current_player[game_indices] ^= 1
        
        # Update hash for new player
        new_players = self.current_player[game_indices].long()
        self.position_hash[game_indices] ^= self.zobrist_turn[new_players]
    
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
        """Get mask of games that have at least one legal move."""
        if legal_moves is None:
            legal_moves = self.legal_moves()
        
        return find_playable_games(legal_moves)
    

    
    def get_current_flat_union_table(self) -> Tensor:
        """Get current flattened union-find table - with debug visualization"""
    

        
        # For batch 0, let's visualize what's in the table
        batch_idx = 0
        board_size = self.board_size
        
        print(f"\n=== Union-Find Table for Batch {batch_idx} ===")
        print(f"Board size: {board_size}×{board_size}")
        print(f"Table shape: {self.flatten_union_find.shape}")
        
        # Show the structure for first few positions
        print("\nTable structure: [color, parent, rank]")
        print("Flat idx | (r,c) | Color | Parent | Rank")
        print("-" * 45)
        
        for flat_idx in range(min(20, board_size * board_size)):
            row = flat_idx // board_size
            col = flat_idx % board_size
            
            color = self.flatten_union_find[batch_idx, flat_idx, 0].item()
            parent = self.flatten_union_find[batch_idx, flat_idx, 1].item()
            rank = self.flatten_union_find[batch_idx, flat_idx, 2].item()
            
            color_str = "Empty" if color == -1 else ("Black" if color == 0 else "White")
            
            print(f"{flat_idx:8d} | ({row},{col}) | {color:3d} | {parent:6d} | {rank:4d}")


    
    # ==================== KO HANDLING ====================
    
    def _apply_ko_restrictions(self, legal: BoardTensor) -> BoardTensor:
        """Apply ko restrictions to legal moves"""
        has_ko = self.ko_points[:, 0] >= 0
        if not has_ko.any():
            return legal
        
        ko_games = has_ko.nonzero(as_tuple=True)[0]
        ko_rows = self.ko_points[ko_games, 0].long()
        ko_cols = self.ko_points[ko_games, 1].long()
        
        legal = legal.clone()
        legal[ko_games, ko_rows, ko_cols] = False
        return legal
    
    def _detect_ko(self, captured_positions: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Detect and set ko points for single stone captures"""
        batch_idx, rows, cols = captured_positions
        
        # Reset ko points
        self.ko_points.fill_(-1)
        
        # Set all potential ko points
        self.ko_points[batch_idx, 0] = rows.to(torch.int8)
        self.ko_points[batch_idx, 1] = cols.to(torch.int8)
        
        # Count captures using scatter_add
        capture_counts = torch.zeros(self.batch_size, dtype=torch.int8, device=self.device)
        capture_counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.int8))
        
        # Clear ko for games with != 1 capture
        invalid_ko_mask = (capture_counts != 1)
        self.ko_points[invalid_ko_mask] = -1
    
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
        
        # Mask finished games
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
    
    # ==================== MOVE EXECUTION ====================
    

    
    def step(self, positions: PositionTensor) -> None:
        """Execute moves for all games - optimized with single active check"""
        
        # SINGLE COMPUTATION of active games at the beginning
        active_mask = ~self.is_game_over()
        if not active_mask.any():
            return  # All games finished
        
        # Get active indices ONCE
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        active_positions = positions[active_indices]
        
        # Update board history and move count
        self._update_board_history_indexed(active_indices)
        self.move_count[active_indices] += 1
        
        self._invalidate_cache()
        
        # Classify moves (only for active games)
        is_pass = is_pass_move(active_positions)
        is_play = ~is_pass
        
        # Update pass counter
        play_indices = active_indices[is_play]
        pass_indices = active_indices[is_pass]
        
        if play_indices.numel() > 0:
            self.pass_count[play_indices] = 0
        
        if pass_indices.numel() > 0:
            self.pass_count[pass_indices] += 1
        
        # Reset ko for active games
        self.ko_points[active_indices] = -1
        
        # Execute moves
        if is_play.any():
            play_positions = active_positions[is_play]
            play_indices = active_indices[is_play]
            self._place_stones(play_positions, play_indices)
        
        self.get_current_flat_union_table()
        
        # Switch turn only for active games
        self._toggle_turn_indexed(active_indices)
    
    def _place_stones(self, positions: PositionTensor, batch_idx: Tensor) -> None:
        """Place stones and handle captures - WITH SUICIDE PREVENTION"""
        rows, cols = positions[:, 0].long(), positions[:, 1].long()
        colors = self.current_player[batch_idx].long()
        
        # Place stones
        self.stones[batch_idx, colors, rows, cols] = True

        
        # UPDATE UNION-FIND TABLE - Just set the color!
        flat_indices = rows * self.board_size + cols
        self.flatten_union_find[batch_idx, flat_indices, 0] = colors.to(torch.int32)
        
        # Update hash
        self._update_hash_for_positions((batch_idx, rows, cols), colors)
        
        # Handle captures FIRST
        captured_any = self._process_captures(batch_idx, rows, cols)
        
        # Check for suicide AFTER captures
        placed_mask = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                                 dtype=torch.bool, device=self.device)
        placed_mask[batch_idx, rows, cols] = True
        
        current_player_stones = self.get_player_stones()
        placed_groups = self._flood_fill(placed_mask, current_player_stones)
        
        group_adjacent = self._count_neighbors(placed_groups.float()) > 0
        group_liberties = group_adjacent & self.empty_mask
        
        no_liberties = torch.zeros(len(batch_idx), dtype=torch.bool, device=self.device)
        for i in range(len(batch_idx)):
            b = batch_idx[i]
            if placed_groups[b].any():
                no_liberties[i] = ~group_liberties[b].any()
        
        suicide_mask = no_liberties & ~captured_any
        
        if suicide_mask.any():
            suicide_idx = suicide_mask.nonzero(as_tuple=True)[0]
            suicide_batch = batch_idx[suicide_idx]
            suicide_rows = rows[suicide_idx]
            suicide_cols = cols[suicide_idx]
            suicide_colors = colors[suicide_idx]
            
            self.stones[suicide_batch, suicide_colors, suicide_rows, suicide_cols] = False
            
            self._update_hash_for_positions(
                (suicide_batch, suicide_rows, suicide_cols), 
                suicide_colors
            )
    
    def _process_captures(self, batch_idx: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        """Remove captured opponent groups"""
        captured_any = torch.zeros(len(batch_idx), dtype=torch.bool, device=self.device)
        
        move_mask = torch.zeros((self.batch_size, self.board_size, self.board_size), 
                               dtype=torch.float32, device=self.device)
        move_mask[batch_idx, rows, cols] = 1.0
        
        neighbors = self._count_neighbors(move_mask) > 0
        opponent_stones = self.get_opponent_stones()
        seeds = neighbors & opponent_stones
        
        if not seeds.any():
            return captured_any
        
        captured = torch.zeros_like(opponent_stones)
        
        for b in range(self.batch_size):
            if not seeds[b].any():
                continue
            
            seed_positions = seeds[b].nonzero(as_tuple=False)
            already_processed = torch.zeros_like(seeds[b], dtype=torch.bool)
            
            for seed_idx in range(len(seed_positions)):
                seed_r = seed_positions[seed_idx, 0]
                seed_c = seed_positions[seed_idx, 1]
                
                if already_processed[seed_r, seed_c]:
                    continue
                
                single_seed = torch.zeros_like(seeds[b])
                single_seed[seed_r, seed_c] = True
                
                current_group = self._flood_fill(
                    single_seed.unsqueeze(0), 
                    opponent_stones[b].unsqueeze(0)
                ).squeeze(0)
                
                already_processed |= current_group
                
                group_adjacent = self._count_neighbors(current_group.unsqueeze(0).float()).squeeze(0) > 0
                group_has_liberties = (group_adjacent & self.empty_mask[b]).any()
                
                if not group_has_liberties:
                    captured[b] |= current_group
                    batch_mask = batch_idx == b
                    captured_any[batch_mask] = True
        
        if captured.any():
            self._remove_captured_stones(captured)
            
        return captured_any
    
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
        
        colors = (1 - self.current_player[batch_idx]).long()
        self.stones[batch_idx, colors, rows, cols] = False
        
        self._update_hash_for_positions((batch_idx, rows, cols), colors)
        
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
    
    
    
    
    
    
    # ==================== TIMING REPORTS ====================
    
    def print_timing_report(self, top_n: int = 30) -> None:
        """Wrapper to call shared timing report."""
        if self.enable_timing:
            from utils.shared import print_timing_report
            print_timing_report(self, top_n)
            
            
   # ==================== History Record====================
   
    def _update_board_history_indexed(self, game_indices: Tensor) -> None:
        """Update board history for specified games"""
        if game_indices.numel() == 0:
            return
        
        move_idx = self.move_count[game_indices].long()
        
        # Get board state for specified games
        black_flat = self.stones[game_indices, 0].flatten(1, 2)
        white_flat = self.stones[game_indices, 1].flatten(1, 2)
        
        board_state = torch.full_like(black_flat, -1, dtype=torch.int8)
        board_state[black_flat] = 0
        board_state[white_flat] = 1
        
        # Advanced indexing to update history
        batch_range = torch.arange(len(game_indices), device=self.device)
        self.board_history[game_indices[batch_range], move_idx[batch_range]] = board_state