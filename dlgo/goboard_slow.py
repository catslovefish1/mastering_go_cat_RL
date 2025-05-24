# ─────────────────────────────────────────────────────────────────────────────
# Minimal Go engine – ko-safe version
# (drop-in replacement for goboard_slow.py or goboard_basic.py)
# ─────────────────────────────────────────────────────────────────────────────

import copy
from dlgo.gotypes import Player, Point        # gotypes.py supplies these

# ────────── 1. Move ──────────
class Move:
    """Any action a player can take: play, pass, or resign."""
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign    # exactly one
        self.point     = point
        self.is_play   = point is not None
        self.is_pass   = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):      return Move(point=point)
    @classmethod
    def pass_turn(cls):        return Move(is_pass=True)
    @classmethod
    def resign(cls):           return Move(is_resign=True)

# ────────── 2. GoString ──────────
class GoString:
    """A chain of same-coloured stones plus its current liberties."""
    def __init__(self, color, stones, liberties):
        self.color      = color
        self.stones     = set(stones)
        self.liberties  = set(liberties)

    def remove_liberty(self, pt): self.liberties.discard(pt)
    def add_liberty(self, pt):    self.liberties.add(pt)

    def merged_with(self, other):
        assert other.color == self.color
        combined   = self.stones | other.stones
        liberties  = (self.liberties | other.liberties) - combined
        return GoString(self.color, combined, liberties)

    @property
    def num_liberties(self): return len(self.liberties)

    # ── NEW: allow structural equality so board snapshots compare cleanly ──
    def __eq__(self, other):
        return (isinstance(other, GoString) and
                self.color     == other.color and
                self.stones    == other.stones and
                self.liberties == other.liberties)

    def __hash__(self):
        return hash((self.color, frozenset(self.stones), frozenset(self.liberties)))

# ────────── 3. Board ──────────
class Board:
    """Handles stone placement, capture, and liberty bookkeeping."""
    def __init__(self, num_rows, num_cols):
        self.num_rows, self.num_cols = num_rows, num_cols
        self._grid = {}                       # Point → GoString

    # helpers ---------------------------------------------------------
    def is_on_grid(self, pt): return 1 <= pt.row <= self.num_rows and 1 <= pt.col <= self.num_cols
    def get(self, pt):        return self._grid.get(pt).color if pt in self._grid else None
    def get_go_string(self, pt): return self._grid.get(pt)

    # core action -----------------------------------------------------
    def place_stone(self, player, point):
        assert self.is_on_grid(point) and self._grid.get(point) is None

        adjacent_same, adjacent_opposite, liberties = [], [], []
        for n in point.neighbors():
            if not self.is_on_grid(n): continue
            neighbor_string = self._grid.get(n)
            if neighbor_string is None:
                liberties.append(n)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same:
                    adjacent_same.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite:
                    adjacent_opposite.append(neighbor_string)

        new_string = GoString(player, [point], liberties)
        for same_string in adjacent_same:
            new_string = new_string.merged_with(same_string)
        for s_pt in new_string.stones:
            self._grid[s_pt] = new_string

        for other_string in adjacent_opposite:
            other_string.remove_liberty(point)
        for other_string in adjacent_opposite:
            if other_string.num_liberties == 0:
                self._remove_string(other_string)

    def _remove_string(self, string):
        """Delete captured string and give liberties back to its neighbours."""
        for pt in string.stones:
            for n in pt.neighbors():
                n_string = self._grid.get(n)
                if n_string is None or n_string is string: continue
                n_string.add_liberty(pt)
            del self._grid[pt]

    # ── NEW: structural equality so ko test can compare boards directly ──
    def __eq__(self, other):
        return (isinstance(other, Board) and
                self.num_rows == other.num_rows and
                self.num_cols == other.num_cols and
                self._grid    == other._grid)

    def __hash__(self):
        return hash((self.num_rows, self.num_cols, frozenset(self._grid.items())))

# ────────── 4. GameState ──────────
class GameState:
    """Board + turn + history; knows how to create the next state."""
    def __init__(self, board, next_player, previous, move):
        self.board           = board
        self.next_player     = next_player
        self.previous_state  = previous
        self.last_move       = move

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    # ---------- game-end / legality helpers ----------
    def is_over(self):
        if self.last_move is None: return False
        second_last = self.previous_state.last_move if self.previous_state else None
        return (self.last_move.is_resign or
                (self.last_move.is_pass and second_last and second_last.is_pass))

    # ── FIXED: compare (player.other, next_board) against history ──
    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)

        past_state = self.previous_state          # start one step back
        while past_state is not None:
            if (past_state.next_player == player.other and
                    past_state.board == next_board):
                return True
            past_state = past_state.previous_state
        return False

    def is_valid_move(self, move):
        if self.is_over():                             return False
        if move.is_pass or move.is_resign:             return True
        if not self.board.is_on_grid(move.point):      return False
        if self.board.get(move.point) is not None:     return False

        # self-capture
        test_state = self.apply_move(move)
        suicide = test_state.board.get_go_string(move.point).num_liberties == 0
        return (not suicide and
                not self.does_move_violate_ko(self.next_player, move))

    # -----------------------------------
    def legal_moves(self):
        """Return a list of every legal move in the current position."""
        moves = []
        for r in range(1, self.board.num_rows + 1):
            for c in range(1, self.board.num_cols + 1):
                p = Point(r, c)
                if self.board.get(p) is None:
                    candidate = Move.play(p)
                    if self.is_valid_move(candidate):
                        moves.append(candidate)
        moves.append(Move.pass_turn())   # pass is always legal
        moves.append(Move.resign())      # resign is always legal
        return moves
