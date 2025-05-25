# ascii_board.py
from dlgo.gotypes import Point, Player

# Unicode stones:
BLACK_STONE  = "●"   # U+25CF, BLACK CIRCLE
WHITE_STONE  = "○"   # U+25CB, WHITE CIRCLE
EMPTY_POINT  = "·"   # U+00B7, MIDDLE DOT  (feel free to change)

def show(board):
    """Print a Go board using Unicode stones (top row first)."""
    for r in range(board.num_rows, 0, -1):
        row = []
        for c in range(1, board.num_cols + 1):
            p = Point(r, c)
            s = board.get(p)
            row.append(
                BLACK_STONE if s == Player.black else
                WHITE_STONE if s == Player.white else
                EMPTY_POINT
            )
        print(" ".join(row))
    print()  # blank line after the diagram
