"""agents/mcts.py - MCTS-based Go agent.

This agent uses Monte Carlo Tree Search for move selection.
It can work with or without a neural network for position evaluation.
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

# Import engine
from engine.tensor_native import TensorBoard

# Import MCTS implementation
from NN.mcts import TinyMCTS, TinyPolicyValueNet

# Import shared utilities
from utils.shared import (
    select_device,
    create_pass_positions
)


class MCTSAgent:
    """MCTS-based agent for playing Go.
    
    This agent uses Monte Carlo Tree Search to select moves.
    It can operate in two modes:
    1. Pure MCTS with random rollouts
    2. Neural network guided MCTS
    
    Attributes:
        device: Computation device (CUDA/MPS/CPU)
        mcts: TinyMCTS instance for tree search
        network: Optional neural network for evaluation
    """
    
    def __init__(
        self,
        simulations: int = 100,
        c_puct: float = 1.0,
        use_network: bool = False,
        network_hidden_dim: int = 64,
        device: Optional[torch.device | str] = None
    ):
        """Initialize MCTS agent.
        
        Args:
            simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for UCB formula
            use_network: Whether to use neural network evaluation
            network_hidden_dim: Hidden dimension for network (if used)
            device: Target device for computations
        """
        self.device = (
            torch.device(device) if device is not None 
            else select_device()
        )
        
        # Create network if requested
        self.network = None
        if use_network:
            # Assume board size will be provided later or use default
            self.network = TinyPolicyValueNet(
                board_size=19,  # Will be updated on first call
                hidden_dim=network_hidden_dim
            ).to(self.device)
            self.network.eval()  # Set to evaluation mode
        
        # Create MCTS instance
        self.mcts = TinyMCTS(
            network=self.network,
            simulations=simulations,
            c_puct=c_puct,
            device=self.device
        )
        
        self.simulations = simulations
        self._initialized_board_size = None
    
    def select_moves(self, boards: TensorBoard, temperature: float = 1.0) -> Tensor:
        """Select moves for all games in batch using MCTS.
        
        Note: This implementation processes games sequentially.
        A production version would use batched MCTS for efficiency.
        
        Args:
            boards: TensorBoard instance with current game states
            temperature: Sampling temperature (0 = greedy, 1 = proportional)
            
        Returns:
            Tensor of shape (B, 2) with [row, col] coordinates.
            Returns [-1, -1] for pass moves.
        """
        batch_size = boards.batch_size
        board_size = boards.board_size
        
        # Initialize network for correct board size if needed
        if self.network is not None and self._initialized_board_size != board_size:
            self.network = TinyPolicyValueNet(
                board_size=board_size,
                hidden_dim=self.network.conv1.out_channels
            ).to(self.device)
            self.network.eval()
            self.mcts.network = self.network
            self._initialized_board_size = board_size
        
        # Initialize moves as passes
        moves = create_pass_positions(batch_size, self.device).to(torch.int32)
        
        # Process each game
        for i in range(batch_size):
            if not boards.is_game_over()[i]:
                # Extract single game state
                single_board = self._extract_single_board(boards, i)
                
                # Run MCTS to select move
                move = self.mcts.select_move(single_board, temperature)
                moves[i] = move.to(torch.int32)
        
        return moves
    
    def _extract_single_board(self, boards: TensorBoard, index: int) -> TensorBoard:
        """Extract a single board from batch for MCTS processing.
        
        Args:
            boards: Batch of boards
            index: Index of board to extract
            
        Returns:
            Single TensorBoard instance
        """
        single_board = TensorBoard(1, boards.board_size, self.device)
        
        # Copy state for specific game
        single_board.stones[0] = boards.stones[index]
        single_board.current_player[0] = boards.current_player[index]
        single_board.position_hash[0] = boards.position_hash[index]
        single_board.ko_points[0] = boards.ko_points[index]
        single_board.pass_count[0] = boards.pass_count[index]
        
        return single_board


# ========================= CONVENIENCE CONSTRUCTORS =========================

def create_pure_mcts_agent(
    simulations: int = 50,
    device: Optional[torch.device | str] = None
) -> MCTSAgent:
    """Create MCTS agent using only random rollouts.
    
    Args:
        simulations: Number of simulations per move
        device: Computation device
        
    Returns:
        MCTSAgent configured for pure MCTS
    """
    return MCTSAgent(
        simulations=simulations,
        use_network=False,
        device=device
    )


def create_neural_mcts_agent(
    simulations: int = 100,
    hidden_dim: int = 128,
    device: Optional[torch.device | str] = None
) -> MCTSAgent:
    """Create MCTS agent with neural network guidance.
    
    Args:
        simulations: Number of simulations per move
        hidden_dim: Hidden dimension for neural network
        device: Computation device
        
    Returns:
        MCTSAgent configured with neural network
    """
    return MCTSAgent(
        simulations=simulations,
        use_network=True,
        network_hidden_dim=hidden_dim,
        device=device
    )


# ========================= DEMO USAGE =========================

if __name__ == "__main__":
    """Quick test of MCTS agent."""
    from engine.tensor_native import TensorBoard
    from interface.ascii import show
    
    # Create a small board for testing
    board = TensorBoard(batch_size=1, board_size=9)
    
    # Create agents
    pure_mcts = create_pure_mcts_agent(simulations=50)
    neural_mcts = create_neural_mcts_agent(simulations=100)
    
    # Test move selection
    print("Testing Pure MCTS agent...")
    move1 = pure_mcts.select_moves(board)
    print(f"Selected move: {move1}")
    
    print("\nTesting Neural MCTS agent...")
    move2 = neural_mcts.select_moves(board)
    print(f"Selected move: {move2}")
    
    # Play a few moves
    print("\nPlaying a few moves...")
    for i in range(5):
        move = pure_mcts.select_moves(board)
        board.step(move)
        print(f"Move {i+1}: {move[0].tolist()}")
    
    # Show final board
    show(board, header="After 5 moves")