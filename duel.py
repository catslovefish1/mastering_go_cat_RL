from __future__ import annotations

"""Bench‑friendly batch self‑play driver.

Changes vs. original `duel_random_vs_random.py`
------------------------------------------------
* progress logging throttled by `LOG_EVERY` (set 0 to silence)
* avoids per‑step `.item()` sync: copies tensor to CPU once per log
* sample‑board printing gated by `NUM_TO_SHOW`
"""

import time
from engine.tensor_go_engine_working import TensorBoard, TensorBatchBot
from interface.ascii import show

# ---------------------------------------------------------------------------
NUM_TO_SHOW = 3       # 0 = skip pretty print entirely
LOG_EVERY  = 200      # moves between progress lines (0 = silent)

# ---------------------------------------------------------------------------

def simulate_batch_games(n: int, size: int):
    print(f"Simulating {n} games on {size}×{size} boards…")
    tic = time.time()

    boards, bot = TensorBoard(n, size), TensorBatchBot()
    max_moves   = size * size * 10
    m = 0

    while not boards.is_game_over().all() and m < max_moves:
        boards.place_stones_batch(bot.select_moves(boards))
        m += 1
        if LOG_EVERY and m % LOG_EVERY == 0:
            finished = int(boards.is_game_over().sum().cpu())   # one sync
            print(f"Move {m:4}: {finished}/{n} finished")

    # ----- summary ----------------------------------------------------
    scores = boards.get_scores().cpu()   # copy once
    bw = int((scores[:, 0] > scores[:, 1]).sum())
    ww = int((scores[:, 1] > scores[:, 0]).sum())
    dr = n - bw - ww

    dt = time.time() - tic
    print(f"\nFinished in {dt:.2f} s  (avg {dt/max(1,m):.3f} s per ply)\n")
    print(f"Black wins : {bw} ({bw / n:.1%})")
    print(f"White wins : {ww} ({ww / n:.1%})")
    print(f"Draws      : {dr} ({dr / n:.1%})")

    # ----- optional board dump ---------------------------------------
    if NUM_TO_SHOW:
        for i in range(min(NUM_TO_SHOW, n)):
            show(boards, header=f"Game {i + 1} – final position", idx=i)


if __name__ == "__main__":
    simulate_batch_games(n=100, size=7)
