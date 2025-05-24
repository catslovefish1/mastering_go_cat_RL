# dlgo/agents/mcts.py
from __future__ import annotations
import math, random
from typing import Dict, Optional

from dlgo.gotypes import Player
from dlgo.goboard_slow import GameState, Move


# ────────────────────────── MCTS tree node ──────────────────────────
class MCTSNode:
    def __init__(self, game_state: GameState,
                 parent: Optional["MCTSNode"],
                 move: Optional[Move]):
        self.game_state = game_state
        self.parent     = parent
        self.move       = move
        self.children: Dict[Move, MCTSNode] = {}
        self.wins   = 0
        self.visits = 0

    # ── expansion ──
    def add_random_child(self) -> "MCTSNode":
        # (1) prefer ordinary plays
        untried = [m for m in self.game_state.legal_moves()
                   if m.is_play and m not in self.children]
        # (2) if exhausted, allow pass / resign
        if not untried:
            untried = [m for m in self.game_state.legal_moves()
                       if m not in self.children]
        move = random.choice(untried)

        next_state = self.game_state.apply_move(move)
        child = MCTSNode(next_state, parent=self, move=move)
        self.children[move] = child
        return child

    # ── stats ──
    def record_win(self, winner: Player):
        self.visits += 1
        if winner == self.game_state.next_player.other:
            self.wins += 1

    def uct_score(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration  = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


# ─────────────────────────── search bot ────────────────────────────
class MCTSBot:
    """Plain UCT Monte-Carlo Tree Search (no neural net)."""

    def __init__(self, num_rollouts: int = 200, c: float = 1.4):
        self.num_rollouts = num_rollouts
        self.c      = c

    # public ---------------------------------------------------------
    def select_move(self, game_state: GameState) -> Move:
        root = MCTSNode(game_state, parent=None, move=None)

        for _ in range(self.num_rollouts):
            leaf = self._select(root)              # 1 selection
            if not leaf.game_state.is_over():      # 2 expansion
                leaf = leaf.add_random_child()
            winner = self._simulate(leaf.game_state)   # 3 simulation
            self._backpropagate(leaf, winner)          # 4 back-prop

        # choose the most-visited child
        best_move = max(root.children.items(),
                        key=lambda kv: kv[1].visits)[0]
        return best_move

    # internals ------------------------------------------------------
    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children and not node.game_state.is_over():
            node = max(node.children.values(),
                        key=lambda n: n.uct_score(node.visits, self.c))
        return node

    @staticmethod
    def _simulate(state: GameState) -> Player:
        """Random rollout that plays real moves; passes only when stuck."""
        while not state.is_over():
            playable = [m for m in state.legal_moves() if m.is_play]
            move = random.choice(playable) if playable else Move.pass_turn()
            state = state.apply_move(move)
        return state.winner()

    @staticmethod
    def _backpropagate(node: MCTSNode, winner: Player):
        while node is not None:
            node.record_win(winner)
            node = node.parent
