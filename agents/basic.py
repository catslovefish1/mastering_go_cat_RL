"""tensor_batch_bot.py - Tensor-native Go agent with random rollouts.

This agent demonstrates tensor-compatible patterns for batch game playing:
- Fully vectorized move selection across multiple games
- Device-aware tensor operations
- Efficient probability sampling without loops
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor

# Assuming TensorBoard follows our refactored interface
from engine.tensor_native import TensorBoard

# Type aliases for clarity
BatchSize = int
BoardHeight = int
BoardWidth = int
MoveTensor = Tensor  # Shape: (B, 2) where 2 = [row, col]
ProbabilityTensor = Tensor  # Shape: (B, H*W) flattened probabilities
MaskTensor = Tensor  # Shape: (B, H, W) boolean mask


# ========================= DEVICE MANAGEMENT =========================

def select_device() -> torch.device:
    """Select best available device with clear priority.
    
    Returns:
        torch.device: MPS > CUDA > CPU in order of preference
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ========================= RANDOM GO AGENT =========================

class TensorBatchBot:
    """Fully vectorized Go bot using random move selection.
    
    This bot demonstrates efficient batch processing patterns:
    - No Python loops over batch dimension
    - Device-aware tensor allocation
    - Vectorized probability sampling
    
    Attributes:
        device: Computation device (CPU/CUDA/MPS)
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
        probabilities = probabilities / row_sums
        
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


# ========================= ALTERNATIVE AGENTS =========================

class WeightedPolicyBot(TensorBatchBot):
    """Bot that uses position-based weights for move selection.
    
    Demonstrates how to extend the base bot with custom policies
    while maintaining tensor-native patterns.
    """
    
    def __init__(
        self, 
        device: Optional[torch.device | str] = None,
        center_bias: float = 2.0
    ):
        """Initialize with device and policy parameters.
        
        Args:
            device: Computation device
            center_bias: How much to prefer center moves (>1 = prefer center)
        """
        super().__init__(device)
        self.center_bias = center_bias
        self._position_weights: Optional[Tensor] = None
    
    def _get_position_weights(self, height: int, width: int) -> Tensor:
        """Get cached position weights favoring center positions.
        
        Args:
            height: Board height
            width: Board width
            
        Returns:
            Shape (H, W) weight tensor
        """
        # Cache weights for efficiency
        if (self._position_weights is None or 
            self._position_weights.shape != (height, width)):
            
            # Create distance from center
            row_coords = torch.arange(height, device=self.device).float()
            col_coords = torch.arange(width, device=self.device).float()
            
            row_center = (height - 1) / 2
            col_center = (width - 1) / 2
            
            row_dist = (row_coords - row_center).abs()
            col_dist = (col_coords - col_center).abs()
            
            # 2D distance map
            row_dist = row_dist.view(-1, 1)
            col_dist = col_dist.view(1, -1)
            distance = (row_dist**2 + col_dist**2).sqrt()
            
            # Convert distance to weights (closer to center = higher weight)
            max_dist = distance.max()
            self._position_weights = 1 + (max_dist - distance) * self.center_bias / max_dist
        
        return self._position_weights
    
    def _compute_uniform_probabilities(self, flat_legal: Tensor) -> ProbabilityTensor:
        """Override to use weighted probabilities instead of uniform.
        
        Args:
            flat_legal: Shape (num_games, H*W) legal moves mask
            
        Returns:
            Weighted probability distribution
        """
        batch_size = flat_legal.shape[0]
        board_size = flat_legal.shape[1]
        height = width = int(board_size ** 0.5)  # Assume square board
        
        # Get position weights and flatten
        weights = self._get_position_weights(height, width)
        flat_weights = weights.view(-1)
        
        # Apply weights only to legal positions
        weighted_legal = flat_legal.float() * flat_weights.unsqueeze(0)
        
        # Normalize
        row_sums = weighted_legal.sum(dim=1, keepdim=True)
        probabilities = weighted_legal / row_sums.clamp(min=1e-8)
        
        return probabilities


# ========================= USAGE EXAMPLE =========================

def example_usage():
    """Demonstrate how to use the tensor-native bot."""
    # Initialize bot on best available device
    bot = TensorBatchBot()
    
    # Or use weighted policy
    weighted_bot = WeightedPolicyBot(center_bias=3.0)
    
    # Create batch of games
    batch_size = 64
    board_size = 19
    games = TensorBoard(
        batch_size=batch_size,
        board_size=board_size,
        device=bot.device
    )
    
    # Play some moves
    for _ in range(10):
        moves = bot.select_moves(games)
        games.step(moves)
        
        # Check for finished games
        finished = games.is_game_over()
        if finished.all():
            break
    
    # Get final scores
    scores = games.compute_scores()
    print(f"Final scores shape: {scores.shape}")  # (64, 2)


if __name__ == "__main__":
    example_usage()