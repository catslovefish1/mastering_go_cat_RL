"""basic.py - Tensor-native Go agent with random rollouts."""

from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor

from engine.tensor_native import TensorBoard
from utils.shared import (
    select_device,
    flat_to_2d,
    create_pass_positions,
    sample_from_mask
)

class TensorBatchBot:
    """Fully vectorized Go bot using random move selection."""
    
    def __init__(self, device: Optional[torch.device | str] = None):
        """Initialize bot with specified or auto-selected device."""
        self.device = torch.device(device) if device else select_device()
    
    def select_moves(self, boards: TensorBoard) -> Tensor:
        """Select moves for all games using uniform random policy.
        
        Returns:
            Tensor of shape (B, 2) with [row, col] coordinates.
            Returns [-1, -1] for pass moves.
        """
        # Get legal moves for active games only
        legal_moves = boards.legal_moves()  # Already filtered by is_game_over()
        batch_size, height, width = legal_moves.shape
        
        # Initialize all as passes
        moves = create_pass_positions(batch_size, self.device).to(torch.int32)
        
        # Find games with legal moves (simplified - no need for separate method)
        games_with_moves = legal_moves.any(dim=(1, 2))
        
        if games_with_moves.any():
            # Get legal moves for playable games and flatten
            playable_legal = legal_moves[games_with_moves]
            flat_legal = playable_legal.view(playable_legal.shape[0], -1)
            
            # Sample moves
            flat_indices = sample_from_mask(flat_legal, num_samples=1)
            
            # Convert to 2D coordinates
            rows, cols = flat_to_2d(flat_indices.to(torch.int32), width)
            moves[games_with_moves] = torch.stack([rows, cols], dim=1)
        
        return moves