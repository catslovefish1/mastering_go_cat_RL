import copy
import itertools
import random
from dlgo.gotypes import Player, Point   # external helper types

# ────────────────────────────────────────────────────────────────
#  Zobrist initialisation
# ────────────────────────────────────────────────────────────────
_ZOBRIST_CACHE = {}           # (rows, cols) → table
_SIDE_TOKEN_CACHE = {}        # (rows, cols) → {Player: int}

def _init_zobrist(num_rows, num_cols):
    """Return (table, side_token) cached for this board size."""
    key = (num_rows, num_cols)
    if key in _ZOBRIST_CACHE:
        return _ZOBRIST_CACHE[key], _SIDE_TOKEN_CACHE[key]

    rng = random.Random(0xC0FFEE)        # deterministic for tests
    table = {(r, c, color): rng.getrandbits(64)
             for r, c, color in itertools.product(
                 range(1, num_rows + 1),
                 range(1, num_cols + 1),
                 (Player.black, Player.white))}
    side_token = {Player.black: rng.getrandbits(64),
                  Player.white: rng.getrandbits(64)}

    _ZOBRIST_CACHE[key]     = table
    _SIDE_TOKEN_CACHE[key]  = side_token
    return table, side_token


# ────────── 1. Move ──────────
class Move:
    """Any action a player can take: play, pass, or resign."""
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point     = point
        self.is_play   = point is not None
        self.is_pass   = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):   return Move(point=point)
    @classmethod
    def pass_turn(cls):     return Move(is_pass=True)
    @classmethod
    def resign(cls):        return Move(is_resign=True)


# ────────── 2. GoString ──────────
class GoString:
    """A chain of same-coloured stones plus its current liberties."""
    def __init__(self, color, stones, liberties):
        self.color     = color
        self.stones    = set(stones)
        self.liberties = set(liberties)

    def remove_liberty(self, pt): self.liberties.discard(pt)
    def add_liberty(self, pt):    self.liberties.add(pt)

    def merged_with(self, other):
        assert other.color == self.color
        combined  = self.stones | other.stones
        liberties = (self.liberties | other.liberties) - combined
        return GoString(self.color, combined, liberties)

    @property
    def num_liberties(self): return len(self.liberties)

    # structural equality for regression tests
    def __eq__(self, other):
        return (isinstance(other, GoString) and
                self.color     == other.color and
                self.stones    == other.stones and
                self.liberties == other.liberties)

    def __hash__(self):
        return hash((self.color,
                     frozenset(self.stones),
                     frozenset(self.liberties)))


# ────────── 3. Board ──────────
class Board:
    """Handles placement, capture, liberties—plus an incremental hash."""
    def __init__(self, num_rows, num_cols):
        self.num_rows, self.num_cols = num_rows, num_cols
        self._grid = {}                          # Point → GoString
        self._zobrist_table, self._side_token = _init_zobrist(num_rows,
                                                              num_cols)
        self._hash = 0                           # 64-bit board hash

    # helpers ----------------------------------------------------
    def is_on_grid(self, pt): return 1 <= pt.row <= self.num_rows and 1 <= pt.col <= self.num_cols
    def get(self, pt):        return self._grid.get(pt).color if pt in self._grid else None
    def get_go_string(self, pt): return self._grid.get(pt)
    @property
    def zobrist(self): return self._hash

    # core action -----------------------------------------------
    def place_stone(self, player, point):
        assert self.is_on_grid(point) and self._grid.get(point) is None

        adjacent_same, adjacent_opposite, liberties = [], [], []
        for n in point.neighbors():
            if not self.is_on_grid(n):
                continue
            string = self._grid.get(n)
            if string is None:
                liberties.append(n)
            elif string.color == player:
                if string not in adjacent_same:
                    adjacent_same.append(string)
            else:
                if string not in adjacent_opposite:
                    adjacent_opposite.append(string)

        new_string = GoString(player, [point], liberties)
        for same_string in adjacent_same:
            new_string = new_string.merged_with(same_string)
        for s_pt in new_string.stones:
            self._grid[s_pt] = new_string
            self._hash ^= self._zobrist_table[(s_pt.row, s_pt.col, player)]

        for other in adjacent_opposite:
            other.remove_liberty(point)
        for other in adjacent_opposite:
            if other.num_liberties == 0:
                self._remove_string(other)

    def _remove_string(self, string):
        """Delete captured stones and restore liberties to neighbours."""
        for pt in string.stones:
            self._hash ^= self._zobrist_table[(pt.row, pt.col, string.color)]
            for n in pt.neighbors():
                n_string = self._grid.get(n)
                if n_string is None or n_string is string:
                    continue
                n_string.add_liberty(pt)
            del self._grid[pt]

    # equality helpers (useful only in tests) -------------------
    def __eq__(self, other):
        return (isinstance(other, Board) and
                self.num_rows == other.num_rows and
                self.num_cols == other.num_cols and
                self._grid    == other._grid)

    def __hash__(self):
        return hash((self.num_rows,
                     self.num_cols,
                     frozenset(self._grid.items())))


# ────────── 4. GameState ──────────
class GameState:
    """Board + player to move + hash history; knows how to create the next state."""
    def __init__(self, board, next_player, previous, move,
                 situation_hash, hash_history):
        self.board          = board
        self.next_player    = next_player
        self.previous_state = previous
        self.last_move      = move
        self.situation_hash = situation_hash      # board hash ⊕ side token
        self.hash_history   = hash_history        # set of past hashes

    # factory ----------------------------------------------------
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        _, side_token = _init_zobrist(*board_size)
        sit_hash = board.zobrist ^ side_token[Player.black]
        return GameState(board, Player.black, None, None,
                         sit_hash, {sit_hash})

    # -----------------------------------------------------------
    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        side_token = next_board._side_token
        next_hash  = next_board.zobrist ^ side_token[self.next_player.other]
        new_hist   = self.hash_history.copy()
        new_hist.add(next_hash)

        return GameState(next_board, self.next_player.other,
                         self, move, next_hash, new_hist)

    # ---------- game-end / legality helpers ----------
    def is_over(self):
        if self.last_move is None:
            return False
        second_last = self.previous_state.last_move if self.previous_state else None
        return (self.last_move.is_resign or
                (self.last_move.is_pass and second_last and second_last.is_pass))

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        table, token = self.board._zobrist_table, self.board._side_token
        next_hash = (self.board.zobrist ^
                     table[(move.point.row, move.point.col, player)] ^
                     token[player.other])
        return next_hash in self.hash_history

    # ---------- suicide check ----------
    def _would_be_self_capture(self, player, point):
        board = self.board
        # free liberty adjacent?
        if any(board.is_on_grid(n) and board.get(n) is None
               for n in point.neighbors()):
            return False
        # capture adjacent enemy string?
        for n in point.neighbors():
            string = board.get_go_string(n)
            if string and string.color != player and string.num_liberties == 1:
                return False
        return True

    def is_valid_move(self, move):
        if self.is_over():                        return False
        if move.is_pass or move.is_resign:        return True
        if not self.board.is_on_grid(move.point): return False
        if self.board.get(move.point) is not None:return False
        if self._would_be_self_capture(self.next_player, move.point):
            return False
        if self.does_move_violate_ko(self.next_player, move):
            return False
        return True

    # -----------------------------------------------------------
    def legal_moves(self):
        moves = []
        for r in range(1, self.board.num_rows + 1):
            for c in range(1, self.board.num_cols + 1):
                p = Point(r, c)
                if self.board.get(p) is None:
                    m = Move.play(p)
                    if self.is_valid_move(m):
                        moves.append(m)
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    # -----------------------------------------------------------
    def winner(self):
        if not self.is_over():
            return None
        black, white = 0, 0
        for string in self.board._grid.values():
            if string.color == Player.black:
                black += 1
            else:
                white += 1
        return Player.black if black > white else Player.white
