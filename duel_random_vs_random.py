# duel_random_vs_random.py
from dlgo.gotypes import Player
from dlgo.goboard_nocopy import GameState
from dlgo.agents.random_tensor_bot import TensorRandomBot
from dlgo.ascii_board import show_many                         # ← only import we need

# ─────────────────────────────────────────────────────────────
def duel(num_games=2000, board_size=7, report_every=10, preview_boards=1):
    black_wins = white_wins = draws = 0
    blk_blk = wht_blk = drw_blk = 0
    final_boards = []            # store first N boards

    for g in range(1, num_games + 1):
        bots = {Player.black: TensorRandomBot(),
                Player.white: TensorRandomBot()}
        state = GameState.new_game(board_size)

        while not state.is_over():
            state = state.apply_move(
                bots[state.next_player].select_move(state)
            )

        if g <= preview_boards:
            final_boards.append(state.board)

        winner, *_ = state.final_result()
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

                # ── pretty-print the captured boards ─────────────────────
    show_many(final_boards,
              title=f"Final boards for the first {preview_boards} games")

    # ── summary ──────────────────────────────────────────────
    print(f"\nResults on a {board_size}×{board_size} board after {num_games} games:")
    print(f"  Black wins : {black_wins} ({black_wins/num_games:6.2%})")
    print(f"  White wins : {white_wins} ({white_wins/num_games:6.2%})")
    print(f"  Draws      : {draws} ({draws/num_games:6.2%})\n")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    duel()
