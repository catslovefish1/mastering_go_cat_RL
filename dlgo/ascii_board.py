# ascii_board.py  – Unicode Go-board renderer
# -------------------------------------------------
from dlgo.gotypes import Point, Player

BLACK_STONE = "●"      # U+25CF  BLACK CIRCLE
WHITE_STONE = "○"      # U+25CB  WHITE CIRCLE
EMPTY_POINT = "·"      # U+00B7  MIDDLE DOT


def board_to_lines(board):
    """Return a list[str] – one line per row (top row first)."""
    rows = []
    for r in range(board.num_rows, 0, -1):            # top → bottom
        row = []
        for c in range(1, board.num_cols + 1):
            p = Point(r, c)
            s = board.get(p)
            row.append(
                BLACK_STONE if s == Player.black else
                WHITE_STONE if s == Player.white else
                EMPTY_POINT
            )
        rows.append(" ".join(row))
    return rows


def show(board, *, header=None, out=print):
    """Pretty-print one board (optionally preceded by a header string)."""
    if header:
        out(header)
    for line in board_to_lines(board):
        out(line)
    out()                         # blank line


def show_many(boards, *, title="Final board positions",
              start_index=1, out=print):
    """Print a numbered list of boards in order."""
    if title:
        out(title + ":")
        out()
    for i, b in enumerate(boards, start_index):
        show(b, header=f"Game {i}", out=out)
