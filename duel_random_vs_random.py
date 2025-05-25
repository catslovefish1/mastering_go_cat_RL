from dlgo.gotypes import Player
from dlgo.goboard_slow import GameState    # or your no-ko-resign variant
from dlgo.agents.random_bot import RandomBot


def duel(num_games=2000, board_size=5, report_every=10):
    """Run `num_games` random games and print a running tally."""
    black_wins = white_wins = draws = 0
    blk_blk = wht_blk = drw_blk = 0        # block counters

    for g in range(1, num_games + 1):
        bots  = {Player.black: RandomBot(), Player.white: RandomBot()}
        state = GameState.new_game(board_size)

        while not state.is_over():
            move  = bots[state.next_player].select_move(state)
            state = state.apply_move(move)

        winner, _, _ = state.final_result()
        if   winner == Player.black:
            black_wins += 1; blk_blk += 1
        elif winner == Player.white:
            white_wins += 1; wht_blk += 1
        else:
            draws      += 1; drw_blk += 1

        if g % report_every == 0:
            print(f"[games {g-report_every+1:>4}-{g:>4}] "
                  f"block → B:{blk_blk:2} W:{wht_blk:2} D:{drw_blk:2} | "
                  f"total → B:{black_wins:3} W:{white_wins:3} D:{draws:3}")
            blk_blk = wht_blk = drw_blk = 0

    print(f"\nResults on a {board_size}×{board_size} board after {num_games} games:")
    print(f"  Black wins : {black_wins} ({black_wins/num_games:6.2%})")
    print(f"  White wins : {white_wins} ({white_wins/num_games:6.2%})")
    print(f"  Draws      : {draws} ({draws/num_games:6.2%})")


if __name__ == "__main__":
    duel()
