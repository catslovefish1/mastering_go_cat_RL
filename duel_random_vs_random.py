from dlgo.gotypes import Player
from dlgo.goboard_slow import GameState
from dlgo.agents.random_bot import RandomBot


def duel(num_games: int = 300, board_size: int = 5) -> None:
    """Play `num_games` purely-random games and print a report every 10 games."""
    black_total = white_total = draw_total = 0
    blk_block = wht_block = drw_block = 0          # per-10-game block counters

    for g in range(1, num_games + 1):
        bots = {Player.black: RandomBot(), Player.white: RandomBot()}
        state = GameState.new_game(board_size)

        while not state.is_over():
            move = bots[state.next_player].select_move(state)
            state = state.apply_move(move)

        winner = state.winner()
        if   winner == Player.black:
            black_total += 1; blk_block += 1
        elif winner == Player.white:
            white_total += 1; wht_block += 1
        else:
            draw_total  += 1; drw_block += 1

        # ---------- 10-game checkpoint ----------
        if g % 10 == 0:
            print(f"[games {g-9:>4}-{g:>4}]  "
                  f"block → B:{blk_block:2} W:{wht_block:2} D:{drw_block:2}   |   "
                  f"cumulative → B:{black_total:3} W:{white_total:3} D:{draw_total:3}")
            blk_block = wht_block = drw_block = 0   # reset for next block

    # ---------- final summary ----------
    print(f"\nFinal results on a {board_size}×{board_size} board after {num_games} games:")
    print(f"  Black wins : {black_total} ({black_total / num_games:6.2%})")
    print(f"  White wins : {white_total} ({white_total / num_games:6.2%})")
    print(f"  Draws      : {draw_total} ({draw_total  / num_games:6.2%})")


if __name__ == "__main__":
    duel()
