"""NN/mcts.py - Minimal Monte Carlo Tree Search for Go.

This module implements a tiny MCTS that:
- Works with TensorBoard for batch evaluation
- Uses shared utilities for consistency
- Supports neural network value/policy guidance
- Maintains simplicity while being functional
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn as nn

# Import engine and utilities
from engine.tensor_native import TensorBoard
from utils.shared import (
    select_device,
    flat_to_2d,
    coords_to_flat,
    create_pass_positions,
    is_pass_move,
    sample_from_mask
)

# ========================= DATA STRUCTURES =========================

@dataclass
class MCTSNode:
    """Single MCTS node representing a board state.
    
    Attributes:
        visits: Number of times visited
        value_sum: Sum of backup values
        prior: Prior probability from policy network
        children: Dict mapping moves to child nodes
        is_expanded: Whether node has been expanded
    """
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 1.0
    children: Dict[Tuple[int, int], MCTSNode] = None
    is_expanded: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        """Average value of node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, parent_visits: int, c_puct: float = 1.0) -> float:
        """Upper confidence bound score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.value + exploration


# ========================= SIMPLE POLICY/VALUE NETWORK =========================

class TinyPolicyValueNet(nn.Module):
    """Minimal neural network for policy and value estimation.
    
    This is a placeholder - in practice you'd use a proper ResNet.
    """
    
    def __init__(self, board_size: int = 19, hidden_dim: int = 64):
        super().__init__()
        self.board_size = board_size
        
        # Simple conv layers
        self.conv1 = nn.Conv2d(5, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(hidden_dim, 1, 1)
        
        # Value head
        self.value_conv = nn.Conv2d(hidden_dim, 1, 1)
        self.value_fc = nn.Linear(board_size * board_size, 1)
    
    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning policy logits and value.
        
        Args:
            features: Shape (B, 5, H, W) from board.extract_features()
            
        Returns:
            policy_logits: Shape (B, H, W)
            value: Shape (B,) in [-1, 1]
        """
        # Feature extraction
        x = torch.relu(self.conv1(features))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Policy head
        policy_logits = self.policy_conv(x).squeeze(1)
        
        # Value head
        value = self.value_conv(x).squeeze(1)
        value = value.view(value.shape[0], -1)
        value = torch.tanh(self.value_fc(value)).squeeze(1)
        
        return policy_logits, value


# ========================= MCTS IMPLEMENTATION =========================

class TinyMCTS:
    """Minimal MCTS implementation for Go.
    
    This version:
    - Uses single game evaluation (not batched for simplicity)
    - Supports neural network guidance
    - Implements basic UCB selection
    """
    
    def __init__(
        self,
        network: Optional[nn.Module] = None,
        simulations: int = 100,
        c_puct: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """Initialize MCTS.
        
        Args:
            network: Optional neural network for evaluation
            simulations: Number of simulations per move
            c_puct: Exploration constant
            device: Computation device
        """
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device or select_device()
        
        # If no network provided, use random rollouts
        self.use_nn = network is not None
        if self.use_nn:
            self.network.eval()
    
    def select_move(self, board: TensorBoard, temperature: float = 1.0) -> Tensor:
        """Select best move using MCTS.
        
        Args:
            board: Current board state (single game)
            temperature: Sampling temperature (0 = greedy)
            
        Returns:
            Move tensor of shape (2,) with [row, col]
        """
        # Create root node
        root = MCTSNode()
        
        # Run simulations
        for _ in range(self.simulations):
            self._simulate(board, root)
        
        # Select move based on visit counts
        return self._choose_move(board, root, temperature)
    
    def _simulate(self, board: TensorBoard, node: MCTSNode) -> float:
        """Run one simulation from node.
        
        Args:
            board: Current board state
            node: Current tree node
            
        Returns:
            Value estimate for position
        """
        # Check if game over
        if board.is_game_over()[0]:
            return self._evaluate_terminal(board)
        
        # Expand if needed
        if not node.is_expanded:
            return self._expand_and_evaluate(board, node)
        
        # Select best child
        move = self._select_child(board, node)
        
        # Make move and recurse
        board_copy = self._copy_board(board)
        board_copy.step(move.unsqueeze(0))
        
        # Get child node
        move_tuple = (move[0].item(), move[1].item())
        child = node.children[move_tuple]
        
        # Recurse and backup
        value = -self._simulate(board_copy, child)
        self._backup(node, value)
        
        return value
    
    def _expand_and_evaluate(self, board: TensorBoard, node: MCTSNode) -> float:
        """Expand node and evaluate position.
        
        Args:
            board: Current board state
            node: Node to expand
            
        Returns:
            Position value estimate
        """
        node.is_expanded = True
        
        # Get legal moves
        legal_moves = board.legal_moves()[0]  # Single game
        
        if self.use_nn:
            # Get policy and value from network
            features = board.extract_features()
            with torch.no_grad():
                policy_logits, value = self.network(features)
            
            # Mask illegal moves and normalize
            policy_logits = policy_logits[0]
            policy_logits[~legal_moves] = -float('inf')
            priors = torch.softmax(policy_logits.flatten(), dim=0)
            
            # Create children with priors
            for pos in legal_moves.nonzero():
                row, col = pos[0].item(), pos[1].item()
                flat_idx = coords_to_flat(
                    torch.tensor([row]), 
                    torch.tensor([col]), 
                    board.board_size
                )[0]
                prior = priors[flat_idx].item()
                node.children[(row, col)] = MCTSNode(prior=prior)
            
            # Add pass move if no legal moves
            if not legal_moves.any():
                node.children[(-1, -1)] = MCTSNode(prior=1.0)
            
            value_estimate = value[0].item()
        else:
            # Random rollout evaluation
            value_estimate = self._random_rollout(board)
            
            # Uniform priors
            for pos in legal_moves.nonzero():
                row, col = pos[0].item(), pos[1].item()
                node.children[(row, col)] = MCTSNode()
            
            if not legal_moves.any():
                node.children[(-1, -1)] = MCTSNode()
        
        self._backup(node, value_estimate)
        return value_estimate
    
    def _select_child(self, board: TensorBoard, node: MCTSNode) -> Tensor:
        """Select best child using UCB.
        
        Args:
            board: Current board state
            node: Parent node
            
        Returns:
            Selected move as tensor
        """
        best_score = -float('inf')
        best_move = None
        
        for move, child in node.children.items():
            score = child.ucb_score(node.visits, self.c_puct)
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move == (-1, -1):
            return create_pass_positions(1, self.device)[0]
        
        return torch.tensor(best_move, device=self.device)
    
    def _choose_move(
        self, 
        board: TensorBoard, 
        root: MCTSNode, 
        temperature: float
    ) -> Tensor:
        """Choose final move based on visit counts.
        
        Args:
            board: Current board
            root: Root node with statistics
            temperature: Sampling temperature
            
        Returns:
            Selected move
        """
        if temperature == 0:
            # Greedy selection
            best_visits = 0
            best_move = None
            for move, child in root.children.items():
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_move = move
        else:
            # Sample based on visit counts
            moves = []
            visits = []
            for move, child in root.children.items():
                moves.append(move)
                visits.append(child.visits)
            
            # Apply temperature and normalize
            visits_tensor = torch.tensor(visits, dtype=torch.float32)
            probs = torch.pow(visits_tensor, 1.0 / temperature)
            probs = probs / probs.sum()
            
            # Sample
            idx = torch.multinomial(probs, 1).item()
            best_move = moves[idx]
        
        if best_move == (-1, -1):
            return create_pass_positions(1, self.device)[0]
        
        return torch.tensor(best_move, device=self.device)
    
    def _backup(self, node: MCTSNode, value: float) -> None:
        """Backup value through node.
        
        Args:
            node: Node to update
            value: Value to backup
        """
        node.visits += 1
        node.value_sum += value
    
    def _evaluate_terminal(self, board: TensorBoard) -> float:
        """Evaluate terminal position.
        
        Args:
            board: Terminal board state
            
        Returns:
            Value from current player perspective
        """
        scores = board.compute_scores()[0]
        black_score = scores[0].item()
        white_score = scores[1].item()
        
        # Simple difference evaluation
        score_diff = black_score - white_score
        
        # Return from current player perspective
        if board.current_player[0] == 0:  # Black
            return 1.0 if score_diff > 0 else -1.0
        else:  # White
            return 1.0 if score_diff < 0 else -1.0
    
    def _random_rollout(self, board: TensorBoard) -> float:
        """Perform random rollout for evaluation.
        
        Args:
            board: Starting position
            
        Returns:
            Terminal value estimate
        """
        board_copy = self._copy_board(board)
        max_moves = 200  # Prevent infinite games
        
        for _ in range(max_moves):
            if board_copy.is_game_over()[0]:
                break
            
            # Random legal move
            legal = board_copy.legal_moves()[0]
            if legal.any():
                flat_legal = legal.flatten().unsqueeze(0)
                flat_idx = sample_from_mask(flat_legal)[0]
                row, col = flat_to_2d(flat_idx.unsqueeze(0), board.board_size)
                move = torch.stack([row[0], col[0]]).unsqueeze(0)
            else:
                move = create_pass_positions(1, self.device)
            
            board_copy.step(move)
        
        return self._evaluate_terminal(board_copy)
    
    def _copy_board(self, board: TensorBoard) -> TensorBoard:
        """Create a copy of the board for simulation.
        
        Args:
            board: Board to copy
            
        Returns:
            New board instance with same state
        """
        new_board = TensorBoard(1, board.board_size, board.device)
        
        # Copy state
        new_board.stones.copy_(board.stones)
        new_board.current_player.copy_(board.current_player)
        new_board.position_hash.copy_(board.position_hash)
        new_board.ko_points.copy_(board.ko_points)
        new_board.pass_count.copy_(board.pass_count)
        
        return new_board


# ========================= MCTS BOT =========================

class MCTSBot:
    """Bot that uses MCTS for move selection."""
    
    def __init__(
        self,
        network: Optional[nn.Module] = None,
        simulations: int = 100,
        device: Optional[torch.device] = None
    ):
        """Initialize MCTS bot.
        
        Args:
            network: Optional neural network
            simulations: Simulations per move
            device: Computation device
        """
        self.device = device or select_device()
        self.mcts = TinyMCTS(network, simulations, device=self.device)
    
    def select_moves(self, boards: TensorBoard) -> Tensor:
        """Select moves for all games in batch.
        
        Note: This implementation processes games sequentially.
        For production, you'd want proper batched MCTS.
        
        Args:
            boards: TensorBoard with game states
            
        Returns:
            Moves tensor of shape (B, 2)
        """
        batch_size = boards.batch_size
        moves = create_pass_positions(batch_size, self.device)
        
        # Process each game separately (not efficient but simple)
        for i in range(batch_size):
            if not boards.is_game_over()[i]:
                # Extract single game
                single_board = TensorBoard(1, boards.board_size, self.device)
                single_board.stones[0] = boards.stones[i]
                single_board.current_player[0] = boards.current_player[i]
                single_board.position_hash[0] = boards.position_hash[i]
                single_board.ko_points[0] = boards.ko_points[i]
                single_board.pass_count[0] = boards.pass_count[i]
                
                # Run MCTS
                move = self.mcts.select_move(single_board, temperature=1.0)
                moves[i] = move
        
        return moves