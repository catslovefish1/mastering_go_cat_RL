"""utils/shared.py - Shared utilities for tensor-based Go implementation.

This module contains common utilities used across the Go engine, agents, and simulation.
All utilities are self-contained and can be imported by any module.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Tuple, Optional

# ========================= DEVICE UTILITIES =========================

def select_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU.
    
    Returns:
        torch.device: The best available device for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ========================= COORDINATE UTILITIES =========================

def flat_to_2d(flat_indices: Tensor, width: int) -> Tuple[Tensor, Tensor]:
    """Convert flat indices to 2D coordinates.
    
    Args:
        flat_indices: Shape (N,) flat position indices
        width: Board width for modulo operation
        
    Returns:
        Tuple of (rows, cols) tensors
    """
    rows = flat_indices // width
    cols = flat_indices % width
    return rows, cols

def coords_to_flat(rows: Tensor, cols: Tensor, width: int) -> Tensor:
    """Convert 2D coordinates to flat indices.
    
    Args:
        rows: Row coordinates
        cols: Column coordinates
        width: Board width
        
    Returns:
        Flat indices tensor
    """
    return rows * width + cols

# ========================= POSITION UTILITIES =========================

def create_pass_positions(batch_size: int, device: torch.device) -> Tensor:
    """Create tensor of pass moves for given batch size.
    
    Pass moves are represented as [-1, -1] in the position tensor.
    
    Args:
        batch_size: Number of positions to create
        device: Target device for tensor creation
        
    Returns:
        Tensor of shape (batch_size, 2) filled with -1
    """
    return torch.full((batch_size, 2), -1, dtype=torch.int16, device=device)

def is_pass_move(positions: Tensor) -> Tensor:
    """Check which positions are pass moves.
    
    Args:
        positions: Shape (B, 2) tensor of [row, col] positions
        
    Returns:
        Shape (B,) boolean tensor - True for pass moves
    """
    return positions[:, 0] < 0

# ========================= GAME STATE UTILITIES =========================

def find_playable_games(legal_moves: Tensor) -> Tensor:
    """Identify which games have at least one legal move.
    
    Args:
        legal_moves: Shape (B, H, W) boolean mask of legal moves
        
    Returns:
        Shape (B,) boolean mask - True where game has legal moves
    """
    # Flatten spatial dimensions and check if any legal move exists
    batch_size = legal_moves.shape[0]
    flat_legal = legal_moves.view(batch_size, -1)
    return flat_legal.any(dim=1)

# ========================= PROBABILITY UTILITIES =========================

def compute_uniform_probabilities(mask: Tensor) -> Tensor:
    """Compute uniform probability distribution over True values in mask.
    
    Args:
        mask: Shape (N, M) boolean mask
        
    Returns:
        Shape (N, M) probability distribution (sums to 1 along dim=1)
    """
    # Convert to float for probability computation
    probabilities = mask.float()
    
    # Normalize each row to sum to 1
    # keepdim=True maintains shape for broadcasting
    row_sums = probabilities.sum(dim=1, keepdim=True)
    safe_sums = row_sums.clamp(min=1.0)  # Avoid division by zero
    probabilities = probabilities / safe_sums
    
    return probabilities

def sample_from_mask(mask: Tensor, num_samples: int = 1) -> Tensor:
    """Sample indices from a boolean mask with uniform probability.
    
    Args:
        mask: Shape (N, M) boolean mask
        num_samples: Number of samples per row
        
    Returns:
        Shape (N, num_samples) or (N,) if num_samples=1
    """
    probabilities = compute_uniform_probabilities(mask)
    sampled = torch.multinomial(probabilities, num_samples=num_samples)
    
    if num_samples == 1:
        sampled = sampled.squeeze(1)
    
    return sampled

# ========================= TENSOR SHAPE UTILITIES =========================

def ensure_4d(tensor: Tensor) -> Tensor:
    """Ensure tensor is 4D by adding singleton dimensions if needed.
    
    Args:
        tensor: Input tensor of shape (H, W) or (B, H, W) or (B, C, H, W)
        
    Returns:
        4D tensor of shape (B, C, H, W)
    """
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)
    return tensor

def ensure_3d(tensor: Tensor) -> Tensor:
    """Ensure tensor is 3D by adding/removing dimensions as needed.
    
    Args:
        tensor: Input tensor
        
    Returns:
        3D tensor of shape (B, H, W)
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(1)
    return tensor

# ========================= BATCH UTILITIES =========================

def get_batch_indices(batch_size: int, device: torch.device) -> Tensor:
    """Create tensor of batch indices [0, 1, 2, ..., batch_size-1].
    
    Args:
        batch_size: Number of indices to create
        device: Target device
        
    Returns:
        Tensor of shape (batch_size,) with indices
    """
    return torch.arange(batch_size, device=device)

def scatter_first_occurrence(
    batch_idx: Tensor, 
    values: Tensor, 
    batch_size: int, 
    default: int = -1
) -> Tensor:
    """For each batch, get the first occurrence of a value.
    
    Useful for finding single captures per batch in Ko detection.
    
    Args:
        batch_idx: Batch indices for each value
        values: Values to scatter
        batch_size: Total number of batches
        default: Default value for batches with no occurrence
        
    Returns:
        Tensor of shape (batch_size,) with first value per batch
    """
    result = torch.full((batch_size,), default, dtype=values.dtype, device=values.device)
    
    # Reverse to ensure first occurrence wins (scatter keeps last)
    reversed_idx = torch.arange(len(batch_idx) - 1, -1, -1, device=batch_idx.device)
    result.scatter_(0, batch_idx[reversed_idx], values[reversed_idx])
    
    return result