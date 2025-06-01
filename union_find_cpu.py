import numpy as np
import pandas as pd

"""Minimal 4 × 4 Go demo using Union‑Find — cleaned valid_neighbor

Tweaks in this revision
───────────────────────
* Dropped the `unflat` helper: now we stay **entirely** in 1‑D space.
* `valid_neighbor` uses `% NCOLS` to block horizontal wrap‑around.
* No other functional changes.
"""

# ───────────── Config ──────────────
NROWS, NCOLS = 4, 4
NEIGH_OFFSETS = np.array([-NCOLS, NCOLS, -1, 1], np.int32)  # ↑ ↓ ← →

# ──────────── Utilities ────────────
flat = lambda r, c: r * NCOLS + c  # 2‑D → 1‑D flat index


def valid_neighbor(f: int, off: int) -> bool:
    """Branch‑free neighbour test (GPU‑friendly).

    Computes a boolean mask with pure arithmetic & bitwise ops so there are
    *no* Python `if` statements.  Still returns a Python bool for clarity.
    """
    col = f % NCOLS      # 0 … NCOLS‑1
    row = f // NCOLS     # 0 … NROWS‑1
    g   = f + off        # candidate neighbour flat index

    # In‑board mask (top/bottom guard)
    on_board = (g >= 0) & (g < NROWS * NCOLS)

    # Horizontal wrap mask: true when move would cross row boundary
    left_wrap  = (off == -1)      & (col == 0)
    right_wrap = (off ==  1)      & (col == NCOLS - 1)

    # Vertical wrap mask: true when move would leave board vertically
    up_wrap    = (off == -NCOLS) & (row == 0)
    down_wrap  = (off ==  NCOLS) & (row == NROWS - 1)

    reject = left_wrap | right_wrap | up_wrap | down_wrap
    return bool(on_board & ~reject)




# ─────────── Union–Find core ──────────

def uf_find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def uf_union(parent: np.ndarray, a: int, b: int) -> None:
    ra, rb = uf_find(parent, a), uf_find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)  # low index wins


# ────────── Game helpers ────────────

def merge_neighbours(parent: np.ndarray, colour: np.ndarray, f: int) -> None:
    for off in NEIGH_OFFSETS:
        if valid_neighbor(f, off):
            g = f + off
            if colour[g] == colour[f]:
                uf_union(parent, f, g)


def add_stone(state, parent, colour, r, c, col):
    f = flat(r, c)
    state[r, c] = col
    colour[f]   = col
    merge_neighbours(parent, colour, f)


def compress_all(parent):
    for i in range(parent.size):
        parent[i] = uf_find(parent, i)


# ───────── Visual helpers ──────────

def show_table(title, parent, colour, liberties):
    tbl = np.stack([np.arange(parent.size), parent, colour, liberties], 1)
    print(f"{title}\n" + pd.DataFrame(
        tbl, columns=["index", "parent", "colour", "lib"]
    ).to_string(index=False) + "\n")


def print_board(state):
    sym = {-1: ".", 0: "●", 1: "○"}
    for row in state:
        print(" ".join(sym[v] for v in row))
    print()


# ────────── Setup helpers ──────────

def init_uf(state):
    N = state.size
    parent    = np.arange(N, dtype=np.int32)
    colour    = np.full(N, -1, np.int32)
    liberties = np.zeros(N, np.int32)
    return parent, colour, liberties


def bootstrap(state, parent, colour):
    for (r, c), v in np.ndenumerate(state):
        if v != -1:
            f = flat(r, c)
            colour[f] = v
            merge_neighbours(parent, colour, f)


# ───────────── Demo ───────────────

def demo():
    state = np.array([
        [-1, 0, -1, -1],
        [ 0, -1, 0,  0],
        [-1, 0, -1, -1],
        [-1, -1, -1, -1],
    ], np.int32)

    parent, colour, liberties = init_uf(state)
    show_table("UF INITIAL", parent, colour, liberties)

    bootstrap(state, parent, colour)
    show_table("UF AFTER BOOTSTRAP", parent, colour, liberties)
    print("Board after bootstrap:")
    print_board(state)

    add_stone(state, parent, colour, 1, 1, 0)  # new black stone (1,1)
    show_table("UF AFTER MOVE", parent, colour, liberties)

    compress_all(parent)
    show_table("UF AFTER COMPRESS", parent, colour, liberties)
    print("Final board:")
    print_board(state)


if __name__ == "__main__":
    demo()
