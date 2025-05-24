from dlgo.gotypes import Player
from dlgo.goboard_slow import GameState
from dlgo.agents import MCTSBot, RandomBot

def duel(num_games=3, board_size=4):
    wins = 0
    for _ in range(num_games):
        bots = {Player.black: MCTSBot(2),  # ← 200 rollouts per move
                Player.white: RandomBot()}
        state = GameState.new_game(board_size)
        while not state.is_over():
            move = bots[state.next_player].select_move(state)
            state = state.apply_move(move)
        if state.winner() == Player.black:
            wins += 1
    print(f"MCTSBot won {wins}/{num_games} games "
          f"({wins/num_games*100:.1f} %)")

if __name__ == "__main__":
    duel()
