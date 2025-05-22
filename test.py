from dlgo.gotypes import Player, Point
from dlgo.goboard_basic import Board

board_size = 9                  # 9×9 beginner board
board = Board(board_size, board_size)

# OPTIONAL: place the very first black stone at (4,4)
board.place_stone(Player.black, Point(4, 4))

# Simple ASCII dump
for row in reversed(range(1, board_size + 1)):
    line = ""
    for col in range(1, board_size + 1):
        stone = board.get(Point(row, col))
        line += (
            "-" if stone is None else
            "X" if stone is Player.black else
            "O"
        )
    print(line)
