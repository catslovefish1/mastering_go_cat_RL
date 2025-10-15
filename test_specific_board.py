# test_specific_board.py
# Verify removal logic only (no ko): BLACK plays at (0,0) and captures entire opponent group.
# Pretty-prints the board before/after.

import torch

# Adjust these imports to match your actual filenames:
from engine.GoLegalMoveChecker import GoLegalMoveChecker
from engine.tensor_native import TensorBoard, Stone   # change module name if needed

SYMBOLS = {
    Stone.EMPTY: ".",
    Stone.BLACK: "X",
    Stone.WHITE: "O",
}

def print_board(board_tensor: torch.Tensor, title: str = ""):
    """
    board_tensor: (1, H, W) int8 on any device
    """
    if title:
        print(f"\n{title}")
    b = board_tensor[0].to("cpu").numpy()  # (H, W)
    H, W = b.shape

    # column header
    col_header = "    " + " ".join(f"{c:2d}" for c in range(W))
    print(col_header)
    print("    " + "--" * W)

    # rows
    for r in range(H):
        row_syms = " ".join(f"{SYMBOLS.get(int(b[r, c]), '?'):>2}" for c in range(W))
        print(f"{r:2d} | {row_syms}")

    print("\nLegend: X=Black, O=White, .=Empty")

def run():
    n = 7
    # 7x7, (0,0) empty, all others white
    flat = [-1] + [1] * (n * n - 1)
    board = torch.tensor(flat, dtype=torch.int8).view(1, n, n)

    # Build engine (no ko), put board on its device
    tb = TensorBoard(batch_size=1, board_size=n, enable_super_ko=False, enable_timing=False,debug_place_trace=True)
    device = tb.device
    tb.board[:] = board.to(device)
    tb.current_player[:] = 0  # black to play

    # Fill capture info (no ko filter because enable_super_ko=False)
    legal_mask = tb.legal_moves()
    assert bool(legal_mask[0, 0, 0].item()), "Black (0,0) must be legal (it captures)."

    print_board(tb.board, title="Before move")

    print("\nCounts before:")
    print("  white stones:", int((tb.board == Stone.WHITE).sum().item()))
    print("  black stones:", int((tb.board == Stone.BLACK).sum().item()))
    print("  empty stones:", int((tb.board == Stone.EMPTY).sum().item()))

    # Play black at (0,0)
    pos = torch.tensor([[0, 0]], dtype=torch.long, device=device)
    tb._place_stones(pos)   # uses _last_capture_info from legal_moves()

    print_board(tb.board, title="After move: Black plays (0,0)")

    print("\nCounts after:")
    print("  white stones:", int((tb.board == Stone.WHITE).sum().item()))
    print("  black stones:", int((tb.board == Stone.BLACK).sum().item()))
    print("  empty stones:", int((tb.board == Stone.EMPTY).sum().item()))
    print("  board[0,0]:  ", int(tb.board[0, 0, 0].item()), "(0 means black)")

if __name__ == "__main__":
    run()
