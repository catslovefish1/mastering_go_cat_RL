from dlgo.gotypes import Point, Player

def show(board):
    for r in range(board.num_rows, 0, -1):
        row = []
        for c in range(1, board.num_cols + 1):
            p = Point(r, c)
            s = board.get(p)
            row.append('B' if s == Player.black
                       else 'W' if s == Player.white
                       else '.')
        print(' '.join(row))
    print()
