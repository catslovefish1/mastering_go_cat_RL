from dlgo.gotypes import Player, Point
from dlgo.goboard_slow import GameState, Move

# 5×5 test game
game = GameState.new_game(5)

def play(point):
    global game
    game = game.apply_move(Move.play(Point(*point)))

# Black surrounds a white stone at (3,3)
play((3,2)); play((2,2))          # B, W
play((2,3)); play((2,4))          # B, W
play((3,4)); play((3,3))          # B, W plays inside
play((4,3))                       # B capture

assert game.board.get(Point(3,3)) is None, "Capture failed!"

print("✓ all good – basic liberties/capture logic passes.")
