"""tensor_native.py - Optimized Go engine. union_find_version
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
        self.register_buffer("move_count", torch.zeros(B, dtype=torch.int16, device=self.device))
    
    def _init_union_find_table(self) -> None:
        """Allocate [colour, parent, liberitry] for every point on the board."""
        B, H, W = self.batch_size, self.board_size, self.board_size
        N_squared = H * W
        

        self.register_buffer(
            "flatten_union_table", 
            torch.zeros((B, N_squared, 4), dtype=torch.int32, device=self.device)
        )
        
        
        # --- column 0 : index --------------------------------------------------
        idx = torch.arange(N_squared, device=self.device, dtype=torch.int32)         # (N²,)
        self.flatten_union_table[..., 0] = idx.unsqueeze(0)                  # broadcast to (B, N²)

        # --- column 1 : parent -------------------------------------------------
        self.flatten_union_table[..., 1] = idx.unsqueeze(0)                                # parent = self

        # --- column 2 : colour -------------------------------------------------
        self.flatten_union_table[..., 2] = -1                                 # empty

        # --- column 3 : liberitry ---------------------------------------------
        self.flatten_union_table[..., 3] = 0                                   # 0 liberties

    # ==================== CORE UTILITIES ====================
    
    def _count_neighbors(self, mask: BoardTensor) -> BoardTensor:
        """Count orthogonal neighbors"""
        result = 1
        return result
    

    def switch_player(self) -> None:
        """Switch current player and update hash for specified games"""
        self.current_player=self.current_player^1
        

    def _invalidate_cache(self) -> None:
        """Clear cached values"""
        self._cache.clear()
    
    # ==================== BOARD QUERIES ====================
    
    
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
    
    
    # ==================== Connected Group  ====================



    # ===  Union–Find helpers  ===============================================

    def _uf_index(self, r: int, c: int) -> int:
        """Flat index (row × N + col)."""
        return r * self.board_size + c


    def _uf_find(self, batch: int, idx: int) -> int:
        """Find set representative with path compression."""
        parent = self.flatten_union_table[batch, idx, 1].item()
        if parent != idx:
            root = self._uf_find(batch, parent)
            self.flatten_union_table[batch, idx, 1] = root  # path-compress
            return root
        return parent


    def _uf_union(self, batch: int, a_idx: int, b_idx: int) -> None:
        """Union two groups (very simple: keep the lower index as root)."""
        root_a = self._uf_find(batch, a_idx)
        root_b = self._uf_find(batch, b_idx)
        
            # ---- combine liberty counters ----------------------------------------
        lib_a = self.flatten_union_table[batch, root_a, 2].item()
        lib_b = self.flatten_union_table[batch, root_b, 2].item()
        merged_lib = lib_a + lib_b          # no overlap because stones touch orthogonally
        if root_a == root_b:
            return
        # Choose root deterministically; rank field (col 2) is kept for later.
        if root_a < root_b:
            self.flatten_union_table[batch, root_b, 1] = root_a
        else:
            self.flatten_union_table[batch, root_a, 1] = root_b


    # ------------ 3. adding a stone -------------------------------------------
    def _union_find_add_stone(self, batch: int, r: int, c: int, colour: int) -> None:
        """
        Put the new stone into the UF table, compute its initial liberties,
        then union with orthogonal neighbours of the same colour.
        """
        flat = self._uf_index(r, c)

        # --- 3.1 initial entry -------------------------------------------------
        self.flatten_union_table[batch, flat, 0] = colour
        self.flatten_union_table[batch, flat, 1] = flat

        # 3.2 count empty orthogonal neighbours for first liberty value
        empty_lib = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                # if target point is empty, increment liberty counter
                if not self.stones[batch, :, nr, nc].any():
                    empty_lib += 1
        self.flatten_union_table[batch, flat, 2] = empty_lib

        # 3.3 merge with same-coloured neighbours --------------------------
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                n_idx = self._uf_index(nr, nc)
                if self.flatten_union_table[batch, n_idx, 0].item() == colour:
                    self._uf_union(batch, flat, n_idx)


    
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
    
    def legal_moves(self,) -> BoardTensor:

        
        # Start with empty positions where no stones are placed
        legal = ~self.stones.any(dim=1)
        
        # Apply ko restrictions
        legal = self._apply_ko_restrictions(legal)
         
        return legal
    

    
    
     # ==================== MOVE EXECUTION ====================
    
    
    
    
    def _place_stones(self, positions: PositionTensor) -> None:
        """Place stones and handle captures - WITH SUICIDE PREVENTION"""
       
        
        
        
    def _process_captures(self, batch_idx: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        """Remove captured opponent groups"""
        
        captured_any = torch.zeros(len(batch_idx), dtype=torch.bool, device=self.device)
        
      
            
        return captured_any
    
    
    
    
        
    
    
    def _flood_fill(self, seeds: BoardTensor, mask: BoardTensor) -> BoardTensor:
        """Expand seeds to complete groups"""
        groups = seeds.clone()

    
        
    
    # ==================== each single STEP ====================
    

    
    def step(self) -> None:
        """Execute moves for all games - optimized with single active check"""
        
 
        player_stones = self.get_player_stones()
        opponent_stones = self.get_opponent_stones()

        
        # Update board history and move count
        self._update_board_history_indexed()
        self.move_count+= 1
        
        # Classify moves (only for active games)
        is_pass = is_pass_move()
        is_play = ~is_pass
        

        
        # Pass-counter update (branch-free)
        self.pass_count = torch.where(
        is_pass,
        self.pass_count + 1,                   # increment where pass
        torch.zeros_like(self.pass_count)      # reset where play
        )
        
        # Reset ko for active games
        self.ko_points.fill(-1)
        

        self._place_stones(PositionTensor)
        
        
        # Switch turn only for active games
        self.switch_player()
    
    
    
   
    
    
    
    

    
    # ==================== GAME STATE ====================
    
    def is_game_over(self) -> BatchTensor:
        """Check if games are finished"""
        return self.pass_count >= 2
    
    def compute_scores(self) -> Tensor:
        """Compute current scores"""
        black = self.stones[:, Stone.BLACK].sum(dim=(1, 2))
        white = self.stones[:, Stone.WHITE].sum(dim=(1, 2))
        return torch.stack([black, white], dim=1)
    
    
    
    
    
    
    # ==================== TIMING REPORTS ====================
    
    def print_cuurent_union_find_table(self,) -> Tensor:
        """Get current flattened union-find table - with debug visualization"""
    

        # For batch 0, let's visualize what's in the table
        batch_idx = 0
        board_size = self.board_size
        
        print(f"\n=== Union-Find Table for Batch {batch_idx} ===")
        print(f"Board size: {board_size}×{board_size}")
        print(f"Table shape: {self.flatten_union_table.shape}")
        
        # Show the structure for first few positions
        print("\nTable structure: [color, parent, rank]")
        print("Flat idx | (r,c) | Color | Parent | Rank")
        print("-" * 45)
        
        for flat_idx in range(board_size * board_size):
            row = flat_idx // board_size
            col = flat_idx % board_size
            
            color = self.flatten_union_table[batch_idx, flat_idx, 0].item()
            parent = self.flatten_union_table[batch_idx, flat_idx, 1].item()
            rank = self.flatten_union_table[batch_idx, flat_idx, 2].item()
            
            
            print(f"{flat_idx:8d} | ({row},{col}) | {color:3d} | {parent:6d} | {rank:4d}")
 
    
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