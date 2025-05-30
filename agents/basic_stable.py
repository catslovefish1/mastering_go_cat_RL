"""basic.py - Tensor-native Go agent with random rollouts.

This agent demonstrates tensor-compatible patterns for batch game playing:
- Fully vectorized move selection across multiple games
- Device-aware tensor operations
- Efficient probability sampling without loops
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor

# Import TensorBoard and select_device from engine
from engine.tensor_native import TensorBoard, select_device

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
        
        # Initialize all moves as passes
        moves = self._create_pass_moves(batch_size)
        
        # Determine which games have legal moves
        games_can_play = self._find_playable_games(legal_moves)
        
        # Select random legal moves for playable games
        if games_can_play.any():
            selected_moves = self._sample_legal_moves(
                legal_moves, games_can_play, width
            )
            moves[games_can_play] = selected_moves
        
        return moves
    
    def _create_pass_moves(self, batch_size: BatchSize) -> MoveTensor:
        """Create tensor of pass moves for all games.
        
        Args:
            batch_size: Number of games in batch
            
        Returns:
            Tensor of shape (B, 2) filled with -1 (pass indicator)
        """
        return torch.full(
            (batch_size, 2), 
            fill_value=-1, 
            dtype=torch.int32, 
            device=self.device
        )
    
    def _find_playable_games(self, legal_moves: MaskTensor) -> Tensor:
        """Identify which games have at least one legal move.
        
        Args:
            legal_moves: Shape (B, H, W) boolean mask
            
        Returns:
            Shape (B,) boolean mask - True where game has legal moves
        """
        # Flatten spatial dimensions and check any legal move exists
        flat_legal = legal_moves.view(legal_moves.shape[0], -1)
        return flat_legal.any(dim=1)
    
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
        # Extract legal moves for playable games: (num_playable, H*W)
        playable_legal = legal_moves[games_can_play]
        flat_legal = playable_legal.view(playable_legal.shape[0], -1)
        
        # Convert to probabilities (uniform distribution over legal moves)
        probabilities = self._compute_uniform_probabilities(flat_legal)
        
        # Sample move indices
        flat_indices = self._sample_from_probabilities(probabilities)
        
        # Convert flat indices to 2D coordinates
        return self._flat_to_coordinates(flat_indices, board_width)
    
    def _compute_uniform_probabilities(
        self, 
        flat_legal: Tensor
    ) -> ProbabilityTensor:
        """Compute uniform probability distribution over legal moves.
        
        Args:
            flat_legal: Shape (num_games, H*W) boolean mask
            
        Returns:
            Shape (num_games, H*W) probability distribution
        """
        # Convert to float for probability computation
        probabilities = flat_legal.float()
        
        # Normalize each row to sum to 1
        # keepdim=True maintains shape for broadcasting
        row_sums = probabilities.sum(dim=1, keepdim=True)
        safe_sums = row_sums.clamp(min=1.0)
        probabilities = probabilities / safe_sums
        
        return probabilities
    
    def _sample_from_probabilities(
        self, 
        probabilities: ProbabilityTensor
    ) -> Tensor:
        """Sample one move per game from probability distributions.
        
        Args:
            probabilities: Shape (num_games, H*W) distributions
            
        Returns:
            Shape (num_games,) flat indices of selected moves
        """
        # Sample 1 index per game according to probabilities
        sampled = torch.multinomial(probabilities, num_samples=1)
        
        # Remove the num_samples dimension: (num_games, 1) -> (num_games,)
        flat_indices = sampled.squeeze(1)
        
        # Convert to int32 for consistency with move tensor dtype
        return flat_indices.to(torch.int32)
    
    def _flat_to_coordinates(
        self, 
        flat_indices: Tensor,
        board_width: BoardWidth
    ) -> MoveTensor:
        """Convert flat indices to 2D board coordinates.
        
        Args:
            flat_indices: Shape (num_games,) flat position indices
            board_width: Width of board for modulo operation
            
        Returns:
            Shape (num_games, 2) tensor of [row, col] coordinates
        """
        rows = flat_indices // board_width
        cols = flat_indices % board_width
        
        return torch.stack([rows, cols], dim=1)