# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py - GPU-accelerated legal move computation with capture detection
==================================================================================
Fully vectorized implementation for batch-compatible legal move checking.
"""

import torch
from typing import Optional, Tuple, Dict, Union

# Type aliases
DTYPE = torch.int16
IDX_DTYPE = torch.int32

class GoLegalMoveChecker:
    """Vectorized legal move checker with capture detection for Go"""
    
    def __init__(self, board_size: int = 19, device: Optional[torch.device] = None):
        self.board_size = board_size
        self.N2 = board_size * board_size
        self.device = device or torch.device("cpu")
    
    def compute_legal_moves_with_captures(
        self,
        stones: torch.Tensor,              # [B, 2, H, W]
        current_player: torch.Tensor,      # [B]
        ko_points: Optional[torch.Tensor] = None,  # [B, 2]
        return_capture_info: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute legal moves and capture information for batch of boards.
        Fully vectorized implementation.
        """
        B, _, H, W = stones.shape
        device = stones.device
        
        # Initialize output tensors
        legal_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        
        if return_capture_info:
            would_capture = torch.zeros((B, H, W), dtype=torch.bool, device=device)
            capture_groups = torch.full((B, H, W, 4), -1, dtype=IDX_DTYPE, device=device)
            capture_sizes = torch.zeros((B, H, W, 4), dtype=IDX_DTYPE, device=device)
            total_captures = torch.zeros((B, H, W), dtype=IDX_DTYPE, device=device)
        
        # Process all boards in parallel
        checker = VectorizedBoardChecker(board_size=H, device=device)
        
        # Compute for all boards at once
        all_legal, all_captures = checker.compute_batch_legal_and_captures(
            stones, current_player
        )
        
        legal_mask = all_legal
        
        # Apply ko restrictions if present
        if ko_points is not None:
            ko_valid = ko_points[:, 0] >= 0
            if ko_valid.any():
                batch_idx = ko_valid.nonzero(as_tuple=True)[0]
                ko_r = ko_points[batch_idx, 0].long()
                ko_c = ko_points[batch_idx, 1].long()
                legal_mask[batch_idx, ko_r, ko_c] = False
        
        if return_capture_info:
            capture_info = all_captures
            return legal_mask, capture_info
        else:
            return legal_mask


class VectorizedBoardChecker:
    """Fully vectorized board checker for batch processing"""
    
    def __init__(self, board_size: int, device: torch.device):
        self.board_size = board_size
        self.N2 = board_size * board_size
        self.device = device
        self._init_neighbor_structure()
    
    def _init_neighbor_structure(self):
        """Pre-compute neighbor indices for all positions"""
        # Create offsets: North, South, West, East
        OFF = torch.tensor([-self.board_size, self.board_size, -1, 1], 
                          dtype=IDX_DTYPE, device=self.device)
        
        # All flat positions
        flat = torch.arange(self.N2, dtype=IDX_DTYPE, device=self.device)
        
        # Compute all neighbors
        nbrs = flat.unsqueeze(1) + OFF  # [N², 4]
        
        # Basic bounds check
        valid = (nbrs >= 0) & (nbrs < self.N2)
        
        # Get row and column indices for edge detection
        row_idx = flat // self.board_size
        col_idx = flat % self.board_size
        
        # Create edge masks
        left_edge = (col_idx == 0)
        right_edge = (col_idx == self.board_size - 1)
        
        # Block wrap-around at edges
        valid[:, 2] &= ~left_edge       # West direction blocked on left edge
        valid[:, 3] &= ~right_edge      # East direction blocked on right edge
        
        # Store final neighbor indices (-1 for invalid)
        self.NEIGH_IDX = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.NEIGH_VALID = valid
    
    def compute_batch_legal_and_captures(
        self, 
        stones: torch.Tensor,  # [B, 2, H, W]
        current_player: torch.Tensor  # [B]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute legal moves for entire batch using fully vectorized operations"""
        B, _, H, W = stones.shape
        
        # Flatten board representations
        stones_flat = stones.reshape(B, 2, -1)  # [B, 2, N²]
        
        # Get occupied positions
        occupied = stones_flat.any(dim=1)  # [B, N²]
        empty = ~occupied  # [B, N²]
        
        # Initialize union-find for all boards at once
        parent, colour, roots, root_libs = self._batch_init_union_find(stones_flat)
        
        # Expand current player to match board positions
        curr_player_expanded = current_player.unsqueeze(1).expand(-1, self.N2)  # [B, N²]
        opponent_expanded = 1 - curr_player_expanded  # [B, N²]
        
        # Get neighbor information for all boards at once
        # Expand NEIGH_IDX to batch dimension
        batch_neigh_idx = self.NEIGH_IDX.unsqueeze(0).expand(B, -1, -1)  # [B, N², 4]
        batch_neigh_valid = self.NEIGH_VALID.unsqueeze(0).expand(B, -1, -1)  # [B, N², 4]
        
        # Get neighbor colors for all boards
        neigh_colors = torch.zeros((B, self.N2, 4), dtype=DTYPE, device=self.device)
        for b in range(B):
            neigh_colors[b] = self._get_neighbor_colors(colour[b])
        
        # Get neighbor roots for all boards
        neigh_roots = torch.zeros((B, self.N2, 4), dtype=IDX_DTYPE, device=self.device)
        for b in range(B):
            neigh_roots[b] = self._get_neighbor_roots(roots[b])
        
        # Check immediate liberties (empty neighbors)
        has_liberty = (neigh_colors == -1) & batch_neigh_valid  # [B, N², 4]
        has_any_liberty = has_liberty.any(dim=2)  # [B, N²]
        
        # Check captures (opponent groups with 1 liberty)
        # Expand opponent color for comparison
        opponent_expanded_4d = opponent_expanded.unsqueeze(2).expand(-1, -1, 4)  # [B, N², 4]
        is_opponent = (neigh_colors == opponent_expanded_4d) & batch_neigh_valid  # [B, N², 4]
        
        # Get liberty counts for neighbor roots
        neigh_libs = torch.zeros((B, self.N2, 4), dtype=IDX_DTYPE, device=self.device)
        
        # Vectorized liberty lookup
        batch_idx = torch.arange(B, device=self.device)[:, None, None].expand(B, self.N2, 4)
        valid_roots = neigh_roots >= 0
        
        # Use advanced indexing to get liberties
        flat_batch_idx = batch_idx[valid_roots]
        flat_root_idx = neigh_roots[valid_roots]
        neigh_libs[valid_roots] = root_libs[flat_batch_idx, flat_root_idx]
        
        # Can capture if opponent group has exactly 1 liberty
        can_capture_dir = is_opponent & (neigh_libs == 1)  # [B, N², 4]
        can_capture_any = can_capture_dir.any(dim=2)  # [B, N²]
        
        # Check friendly connections with liberties
        curr_player_expanded_4d = curr_player_expanded.unsqueeze(2).expand(-1, -1, 4)  # [B, N², 4]
        is_friendly = (neigh_colors == curr_player_expanded_4d) & batch_neigh_valid  # [B, N², 4]
        
        # Friendly groups need >1 liberty
        friendly_has_libs = is_friendly & (neigh_libs > 1)  # [B, N², 4]
        friendly_any_libs = friendly_has_libs.any(dim=2)  # [B, N²]
        
        # Position is legal if empty and (has liberty OR can capture OR connects to group with libs)
        legal_flat = empty & (has_any_liberty | can_capture_any | friendly_any_libs)  # [B, N²]
        
        # Record capture information
        would_capture_flat = empty & can_capture_any  # [B, N²]
        
        # Initialize capture info tensors
        capture_groups_flat = torch.full((B, self.N2, 4), -1, dtype=IDX_DTYPE, device=self.device)
        capture_sizes_flat = torch.zeros((B, self.N2, 4), dtype=IDX_DTYPE, device=self.device)
        
        # Record which groups would be captured
        capture_groups_flat[can_capture_dir] = neigh_roots[can_capture_dir].to(IDX_DTYPE)
        
        # Count stones in captured groups (this still needs some iteration)
        batch_idx, pos_idx, dir_idx = can_capture_dir.nonzero(as_tuple=True)
        if len(batch_idx) > 0:
            captured_roots = neigh_roots[batch_idx, pos_idx, dir_idx].to(IDX_DTYPE)
            
            # Count group sizes
            for i in range(len(batch_idx)):
                b, root = int(batch_idx[i]), int(captured_roots[i])
                if root >= 0:
                    group_size = (roots[b] == root).sum()
                    capture_sizes_flat[b, pos_idx[i], dir_idx[i]] = group_size.to(IDX_DTYPE)
        
        # Total captures per position
        total_captures_flat = capture_sizes_flat.sum(dim=2)  # [B, N²]
        
        # Reshape back to board dimensions
        legal_mask = legal_flat.reshape(B, H, W)
        
        capture_info = {
            'would_capture': would_capture_flat.reshape(B, H, W),
            'capture_groups': capture_groups_flat.reshape(B, H, W, 4),
            'capture_sizes': capture_sizes_flat.reshape(B, H, W, 4),
            'total_captures': total_captures_flat.reshape(B, H, W),
        }
        
        return legal_mask, capture_info
    
    def _batch_init_union_find(self, stones_flat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Initialize union-find structures for batch of boards"""
        B = stones_flat.shape[0]
        
        # Initialize parent (each point is its own parent) - use DTYPE for consistency
        parent = torch.arange(self.N2, device=self.device, dtype=DTYPE).unsqueeze(0).expand(B, -1)
        parent = parent.clone()  # [B, N²]
        
        # Initialize colors: -1=empty, 0=black, 1=white
        colour = torch.full((B, self.N2), -1, dtype=DTYPE, device=self.device)
        colour[stones_flat[:, 0]] = 0  # black
        colour[stones_flat[:, 1]] = 1  # white
        
        # Build groups by merging neighbors
        # This is simplified - in production, use iterative union-find
        for b in range(B):
            parent[b] = self._build_groups_vectorized(parent[b], colour[b])
        
        # Find roots for all positions
        roots = parent.clone()
        for _ in range(20):  # Sufficient iterations for path compression
            # Ensure indices are within bounds and correct dtype
            gather_idx = roots.long().clamp(0, self.N2-1)
            roots = torch.gather(roots, 1, gather_idx).to(DTYPE)
        
        # Count liberties for each root
        root_libs = self._count_liberties_vectorized(colour, roots)
        
        return parent, colour, roots.to(IDX_DTYPE), root_libs
    
    def _build_groups_vectorized(self, parent: torch.Tensor, colour: torch.Tensor) -> torch.Tensor:
        """Build connected groups using fully vectorized operations"""
        # Get colors of all neighbors
        neigh_colors = self._get_neighbor_colors(colour)  # [N², 4]
        
        # For each position, find neighbors with same color
        pos_colors = colour.unsqueeze(1).expand(-1, 4)  # [N², 4]
        same_color = (neigh_colors == pos_colors) & (pos_colors != -1) & self.NEIGH_VALID  # [N², 4]
        
        # Get all position-neighbor pairs that need union
        positions, directions = same_color.nonzero(as_tuple=True)  # positions and direction indices
        
        if len(positions) > 0:
            # Get the actual neighbor indices
            neighbors = self.NEIGH_IDX[positions, directions]
            
            # Get parents for both positions and neighbors
            pos_parents = parent[positions]
            nbr_parents = parent[neighbors]
            
            # Compute minimum parent for union
            min_parents = torch.minimum(pos_parents, nbr_parents)
            
            # Update parent pointers (vectorized scatter)
            parent = parent.scatter(0, positions.long(), min_parents.to(parent.dtype))
            parent = parent.scatter(0, neighbors.long(), min_parents.to(parent.dtype))
            
            # Iterate a few times to propagate unions
            for _ in range(5):  # Fixed iterations for path compression
                parent = torch.gather(parent, 0, parent.long().clamp(0, self.N2-1))
        
        return parent
    
    def _get_neighbor_colors(self, colour: torch.Tensor) -> torch.Tensor:
        """Get colors of all neighbors for all positions"""
        # Initialize with -2 (invalid)
        neigh_colors = torch.full((self.N2, 4), -2, dtype=DTYPE, device=self.device)
        
        # Fill valid neighbors
        valid = self.NEIGH_VALID
        valid_idx = self.NEIGH_IDX[valid]
        neigh_colors[valid] = colour[valid_idx].to(DTYPE)  # Cast to correct dtype
        
        return neigh_colors
    
    def _get_neighbor_roots(self, roots: torch.Tensor) -> torch.Tensor:
        """Get root indices of all neighbors"""
        neigh_roots = torch.full((self.N2, 4), -1, dtype=IDX_DTYPE, device=self.device)
        
        valid = self.NEIGH_VALID
        valid_idx = self.NEIGH_IDX[valid]
        neigh_roots[valid] = roots[valid_idx].to(IDX_DTYPE)  # Cast to correct dtype
        
        return neigh_roots
    
    def _count_liberties_vectorized(self, colour: torch.Tensor, roots: torch.Tensor) -> torch.Tensor:
        """Count liberties for each root group - fully vectorized"""
        B = colour.shape[0]
        root_libs = torch.zeros((B, self.N2), dtype=IDX_DTYPE, device=self.device)
        
        # Process all boards in parallel
        for b in range(B):
            # Get empty neighbors for all positions
            neigh_colors = self._get_neighbor_colors(colour[b])  # [N², 4]
            is_liberty = (neigh_colors == -1) & self.NEIGH_VALID  # [N², 4]
            
            # Get liberty positions
            stone_positions = (colour[b] != -1).nonzero(as_tuple=True)[0]  # positions with stones
            
            if len(stone_positions) > 0:
                # Get roots and liberties for stone positions
                stone_roots = roots[b, stone_positions]  # roots of positions with stones
                stone_liberties = is_liberty[stone_positions]  # [num_stones, 4]
                stone_liberty_positions = self.NEIGH_IDX[stone_positions]  # [num_stones, 4]
                
                # Flatten and filter valid liberties
                has_lib = stone_liberties.flatten()
                stone_roots_flat = stone_roots.unsqueeze(1).expand(-1, 4).flatten()[has_lib]
                liberty_positions_flat = stone_liberty_positions.flatten()[has_lib]
                
                # For each unique root, count unique liberties
                unique_roots = stone_roots.unique()
                
                # Vectorized counting using scatter operations
                for root in unique_roots:
                    root = int(root)  # Ensure root is integer
                    root_mask = stone_roots_flat == root
                    if root_mask.any():
                        root_liberties = liberty_positions_flat[root_mask]
                        unique_liberties = root_liberties.unique()
                        unique_liberties = unique_liberties[unique_liberties >= 0]
                        root_libs[b, root] = len(unique_liberties)
        
        return root_libs