# interface/ascii.py  –  tiniest working version
BLACK_STONE = "●"      # U+25CF
WHITE_STONE = "○"      # U+25CB
EMPTY_POINT = "·"      # U+00B7


def board_to_lines(board, idx: int = 0) -> list[str]:
    """Return the board (idx) of a TensorBoard as list[str]."""
    b = board.stones[idx, 0].cpu()
    w = board.stones[idx, 1].cpu()
    N = board.board_size

    lines = []
    for r in range(N - 1, -1, -1):            # top → bottom
        row = [
            BLACK_STONE if b[r, c] else
            WHITE_STONE if w[r, c] else
            EMPTY_POINT
            for c in range(N)
        ]
        lines.append(" ".join(row))
    return lines


def show(board, *, header: str | None = None,
         idx: int = 0, out=print) -> None:
    """Pretty-print one board (idx) from a TensorBoard batch."""
    if header:
        out(header)
    for line in board_to_lines(board, idx):
        out(line)
    out()            # blank line
