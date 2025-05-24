import random
from dlgo.goboard_slow import Move

class RandomBot:
    """Uniformly chooses a legal move; if no play left, passes."""
    def select_move(self, game_state):
        playable = [m for m in game_state.legal_moves() if m.is_play]
        if not playable:
            return Move.pass_turn()
        return random.choice(playable)
