from dlgo.gotypes    import Player
from dlgo.goboard_slow import GameState, Move
from dlgo.agents.random_bot import RandomBot
from dlgo.ascii_board import show

BOARD_SIZE = 3
bots   = {Player.black: RandomBot(), Player.white: RandomBot()}
state  = GameState.new_game(BOARD_SIZE)
move_no = 1

while not state.is_over():                # ← only condition you need
    player = state.next_player
    bot    = bots[player]

    move   = bot.select_move(state)
    state  = state.apply_move(move)

    print(f'{move_no:3d}. {player.name:5s} ->',
          'pass' if move.is_pass else move.point)
    show(state.board)
    move_no += 1

print("Game over!")