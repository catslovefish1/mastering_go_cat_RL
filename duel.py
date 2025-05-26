# duel_random_vs_random.py
from __future__ import annotations

from engine.tensor_go_engine import TensorBoard, TensorBatchBot
from interface.ascii import show          # Unicode board printer

NUM_TO_SHOW = 3        # how many final boards to display (0 = none)


def simulate_batch_games(n: int = 20, size: int = 7):
    print(f"Simulating {n} games on {size}×{size} boards…")

    boards, bot = TensorBoard(n, size), TensorBatchBot()
    max_moves   = size * size * 2
    m = 0

    while not boards.is_game_over().all() and m < max_moves:
        boards.place_stones_batch(bot.select_moves(boards))
        m += 1
        if m % 50 == 0:
            finished = boards.is_game_over().sum().item()
            print(f"Move {m}: {finished}/{n} finished")

    # ─── summary ───────────────────────────────────────────────────────
    scores = boards.get_scores()
    bw = (scores[:, 0] > scores[:, 1]).sum().item()
    ww = (scores[:, 1] > scores[:, 0]).sum().item()
    dr = n - bw - ww

    print("\nResults")
    print(f"Black wins : {bw} ({bw / n:.1%})")
    print(f"White wins : {ww} ({ww / n:.1%})")
    print(f"Draws      : {dr} ({dr / n:.1%})")

    # ─── show some boards for sanity check ─────────────────────────────
    for i in range(min(NUM_TO_SHOW, n)):
        header = f"Game {i + 1} – final position"
        show(boards, header=header, idx=i)


if __name__ == "__main__":
    simulate_batch_games(200, 5)
