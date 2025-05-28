"""basic.py - Tensor-native Go agent with random rollouts.

This agent demonstrates tensor-compatible patterns for batch game playing:
- Fully vectorized move selection across multiple games
- Device-aware tensor operations
- Efficient probability sampling without loops
- Uses shared utilities from utils.shared module
"""

from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor

# Import TensorBoard from engine
from engine.tensor_native import TensorBoard

# Import shared utilities
from utils.shared import (
    select_device,
    flat_to_2d,
    create_pass_positions,
    compute_uniform_probabilities,
    sample_from_mask
)

# Type aliases for clarity
BatchSize = int
BoardHeight = int
BoardWidth = int
MoveTensor = Tensor  # Shape: (B, 2) where 2 = [row, col]
ProbabilityTensor = Tensor  # Shape: (B, H*W) flattened probabilities
MaskTensor = Tensor  # Shape: (B, H, W) boolean mask

# ========================= RANDOM GO AGENT =========================

class TensorBatchBot:
    """Fully vectorized Go bot using random move selection.
  
    Attributes:
        device: Computation device (CUDA/MPS/CPU)
    """
    
    def __init__(self, device: Optional[torch.device | str] = None):
        """Initialize bot with specified or auto-selected device.
        
        Args:
            device: Target device for computations. If None, auto-selects best.
        """
        self.device = (
            torch.device(device) if device is not None 
            else select_device()
        )
    
    def select_moves(self, boards: TensorBoard) -> MoveTensor:
        """Select moves for all games in batch using uniform random policy.
        
        This method demonstrates key tensor patterns:
        1. Batch-wise legal move checking
        2. Flattened view for probability computation
        3. Conditional move selection (play vs pass)
        4. Vectorized sampling without loops
        
        Args:
            boards: TensorBoard instance with current game states
            
        Returns:
            MoveTensor of shape (B, 2) with [row, col] coordinates.
            Returns [-1, -1] for pass moves.
        """
        # Get legal moves mask: (B, H, W)
        legal_moves = boards.legal_moves()
        batch_size, height, width = legal_moves.shape
        
        # Initialize all moves as passes using shared utility
        moves = create_pass_positions(batch_size, self.device).to(torch.int32)
        
        # Determine which games have legal moves using shared method
        games_can_play = boards.get_playable_games(legal_moves)
        
        # Select random legal moves for playable games
        if games_can_play.any():
            selected_moves = self._sample_legal_moves(
                legal_moves, games_can_play, width
            )
            moves[games_can_play] = selected_moves
        
        return moves
    
    def _sample_legal_moves(
        self, 
        legal_moves: MaskTensor,
        games_can_play: Tensor,
        board_width: BoardWidth
    ) -> MoveTensor:
        """Sample random legal moves for specified games.
        
        Args:
            legal_moves: Shape (B, H, W) full legal moves mask
            games_can_play: Shape (B,) mask of games to process
            board_width: Width of board for coordinate conversion
            
        Returns:
            Shape (num_playable, 2) tensor of [row, col] coordinates
        """
        # Extract legal moves for playable games and flatten
        playable_legal = legal_moves[games_can_play]
        flat_legal = playable_legal.view(playable_legal.shape[0], -1)
        
        # Sample move indices using shared utility
        flat_indices = sample_from_mask(flat_legal, num_samples=1)
        
        # Convert flat indices to 2D coordinates using shared utility
        rows, cols = flat_to_2d(flat_indices.to(torch.int32), board_width)
        return torch.stack([rows, cols], dim=1)