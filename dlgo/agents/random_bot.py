
# dlgo/random_bot.py
# ---------------------------------------------------------------------
# A very small “uniform random” Go bot.
#
# By selecting exclusively from GameState.legal_moves() it automatically
# obeys all rule checks implemented by the engine, including:
#   • ko prohibition (does_move_violate_ko)
#   • self-capture prevention
#
# If no playable moves remain, the bot passes.
# ---------------------------------------------------------------------

import random
from dlgo.goboard_slow import Move


class RandomBot:
    """Uniformly chooses a legal *play* move; if none exist, passes."""

    def select_move(self, game_state):
        # GameState.legal_moves() already filters out illegal moves.
        playable_moves = [m for m in game_state.legal_moves() if m.is_play]

        # No stones can be played: pass the turn.
        if not playable_moves:
            return Move.pass_turn()

        return random.choice(playable_moves)
