"""
duel.py – run N purely-random Go games and report the result.

Requires:
    • gotypes.Player, goboard_slow.GameState  – your minimal engine
    • RandomBot                              – uniform-random move generator
"""

from dlgo.gotypes import Player
from dlgo.goboard_slow import GameState
from dlgo.agents.random_bot import RandomBot


def duel(num_games: int = 300, board_size: int = 5) -> None:
    """Play `num_games` games on a `board_size`×`board_size` board."""
    black_wins = white_wins = draws = 0

    for g in range(1, num_games + 1):
        bots = {
            Player.black: RandomBot(),
            Player.white: RandomBot(),
        }
        state = GameState.new_game(board_size)

        while not state.is_over():
            move = bots[state.next_player].select_move(state)
            state = state.apply_move(move)

        result = state.winner()
        if   result == Player.black:
            black_wins += 1
        elif result == Player.white:
            white_wins += 1
        else:
            draws += 1

    # ---------- final report ----------
    print(f"\nResults on a {board_size}×{board_size} board "
          f"after {num_games} purely-random games:")
    print(f"  Black wins : {black_wins}  "
          f"({black_wins / num_games:6.2%})")
    print(f"  White wins : {white_wins}  "
          f"({white_wins / num_games:6.2%})")
    print(f"  Draws      : {draws}  "
          f"({draws / num_games:6.2%})")


if __name__ == "__main__":
    duel()
