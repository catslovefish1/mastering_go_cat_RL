"""goboard_fast.py – Go board + game state with copy‑on‑write boards and
constant‑time ko detection (Zobrist hashing).  No `deepcopy`, so it stays
fast even over hundreds of games.

Drop this file in **dlgo/goboard_fast.py** (or adjust your imports) and the
rest of the codebase will work unchanged.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Set

from dlgo.gotypes import Player, Point  # external helper types

# ────────────────────────────────────────────────────────────────
#  Zobrist initialisation – one lazily‑filled table per board size
# ────────────────────────────────────────────────────────────────
_ZOBRIST_CACHE: Dict[tuple[int, int], Dict[tuple[int, int, Player], int]] = {}
_SIDE_TOKEN_CACHE: Dict[tuple[int, int], Dict[Player, int]] = {}


def _init_zobrist(num_rows: int, num_cols: int):
    """Return (table, side_token) cached for this board size."""
    key = (num_rows, num_cols)
    if key in _ZOBRIST_CACHE:
        return _ZOBRIST_CACHE[key], _SIDE_TOKEN_CACHE[key]

    rng = random.Random(0xC0FFEE)  # determinism for unit tests
    table = {(r, c, color): rng.getrandbits(64)
             for r, c, color in itertools.product(
                 range(1, num_rows + 1),
                 range(1, num_cols + 1),
                 (Player.black, Player.white))}
    side_token = {Player.black: rng.getrandbits(64),
                  Player.white: rng.getrandbits(64)}

    _ZOBRIST_CACHE[key] = table
    _SIDE_TOKEN_CACHE[key] = side_token
    return table, side_token


# ────────── 1. Move ──────────
class Move:
    """Any action a player can take: play, pass, or resign."""
    __slots__ = ("point", "is_play", "is_pass", "is_resign")

    def __init__(self, point: Optional[Point] = None, *,
                 is_pass: bool = False, is_resign: bool = False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = point is not None
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point: Point) -> "Move":
        return cls(point=point)

    @classmethod
    def pass_turn(cls) -> "Move":
        return cls(is_pass=True)

    @classmethod
    def resign(cls) -> "Move":
        return cls(is_resign=True)

    # readable repr for debugging / tests
    def __repr__(self):
        if self.is_play:
            return f"Move.play({self.point})"
        if self.is_pass:
            return "Move.pass"
        return "Move.resign"


# ────────── 2. GoString ──────────
class GoString:
    """A chain of same‑coloured stones plus its current liberties."""
    __slots__ = ("color", "stones", "liberties")

    def __init__(self, color: Player, stones: Set[Point],
                 liberties: Set[Point]):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)

    def remove_liberty(self, pt: Point):
        self.liberties.discard(pt)

    def add_liberty(self, pt: Point):
        self.liberties.add(pt)

    def merged_with(self, other: "GoString") -> "GoString":
        assert other.color == self.color
        combined = self.stones | other.stones
        liberties = (self.liberties | other.liberties) - combined
        return GoString(self.color, combined, liberties)

    @property
    def num_liberties(self) -> int:
        return len(self.liberties)

    # structural equality for regression tests
    def __eq__(self, other):
        return (isinstance(other, GoString) and
                self.color == other.color and
                self.stones == other.stones and
                self.liberties == other.liberties)

    def __hash__(self):  # for use as dict key
        return hash((self.color, frozenset(self.stones),
                     frozenset(self.liberties)))


# ────────── 3. Board ──────────
class Board:
    """Handles placement, capture, liberty bookkeeping – plus an incremental hash."""

    __slots__ = ("num_rows", "num_cols", "_grid", "_zobrist_table",
                 "_side_token", "_hash")

    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows, self.num_cols = num_rows, num_cols
        self._grid: Dict[Point, GoString] = {}
        self._zobrist_table, self._side_token = _init_zobrist(num_rows,
                                                              num_cols)
        self._hash: int = 0  # 64‑bit incremental hash of the board position

    # ── helpers ────────────────────────────────────────────────
    def is_on_grid(self, pt: Point) -> bool:
        return 1 <= pt.row <= self.num_rows and 1 <= pt.col <= self.num_cols

    def get(self, pt: Point) -> Optional[Player]:
        string = self._grid.get(pt)
        return string.color if string else None

    def get_go_string(self, pt: Point) -> Optional[GoString]:
        return self._grid.get(pt)

    @property
    def zobrist(self) -> int:
        return self._hash

    # ── main action ────────────────────────────────────────────
    def place_stone(self, player: Player, point: Point):
        """Mutates the board in‑place *and* keeps the Zobrist hash in sync."""
        assert self.is_on_grid(point), "Point off board"
        assert point not in self._grid, "Point already occupied"

        adjacent_same: List[GoString] = []
        adjacent_opposite: List[GoString] = []
        liberties: List[Point] = []

        for n in point.neighbors():
            if not self.is_on_grid(n):
                continue
            string = self._grid.get(n)
            if string is None:
                liberties.append(n)
            elif string.color == player:
                if string not in adjacent_same:
                    adjacent_same.append(string)
            else:  # opponent
                if string not in adjacent_opposite:
                    adjacent_opposite.append(string)

        new_string = GoString(player, {point}, set(liberties))
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

    def _remove_string(self, string: GoString):
        """Delete captured stones and restore liberties to neighbours."""
        for pt in string.stones:
            self._hash ^= self._zobrist_table[(pt.row, pt.col, string.color)]
            del self._grid[pt]
            for n in pt.neighbors():
                n_string = self._grid.get(n)
                if n_string and n_string is not string:
                    n_string.add_liberty(pt)

    # ── cheap structural copy (no deepcopy!) ───────────────────
    def copy(self) -> "Board":
        """Return a board that shares *no mutable state* with this one but
        reuses the (immutable) Zobrist tables.  O(number of strings)."""
        new_board = Board(self.num_rows, self.num_cols)
        new_board._zobrist_table = self._zobrist_table  # share read‑only data
        new_board._side_token = self._side_token
        new_board._hash = self._hash

        # Clone each GoString once and re‑wire grid references.
        clone: Dict[GoString, GoString] = {}
        for pt, string in self._grid.items():
            if string not in clone:
                clone[string] = GoString(string.color,
                                         string.stones.copy(),
                                         string.liberties.copy())
            new_board._grid[pt] = clone[string]
        return new_board

    # ── equality helpers (useful only in tests) ────────────────
    def __eq__(self, other):
        return (isinstance(other, Board) and
                self.num_rows == other.num_rows and
                self.num_cols == other.num_cols and
                self._grid == other._grid)

    def __hash__(self):
        return hash((self.num_rows, self.num_cols,
                     frozenset(self._grid.items())))


# ────────── 4. GameState ──────────
class GameState:
    """Immutable game node: (board, side‑to‑play, ko‑hash history)."""
    __slots__ = ("board", "next_player", "previous_state", "last_move",
                 "situation_hash", "hash_history")

    def __init__(self, board: Board, next_player: Player,
                 previous: Optional["GameState"], move: Optional[Move],
                 situation_hash: int, hash_history: Set[int]):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        self.situation_hash = situation_hash  # board hash ⊕ side token
        self.hash_history = hash_history

    # ── factory ────────────────────────────────────────────────
    @classmethod
    def new_game(cls, board_size: int | tuple[int, int]):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        _, side_token = _init_zobrist(*board_size)
        sit_hash = board.zobrist ^ side_token[Player.black]
        return cls(board, Player.black, None, None, sit_hash, {sit_hash})

    # ── move play ──────────────────────────────────────────────
    def apply_move(self, move: Move) -> "GameState":
        if move.is_play:
            next_board = self.board.copy()
            next_board.place_stone(self.next_player, move.point)  # mutates copy
        else:
            next_board = self.board  # pass / resign carries board forward

        side_token = next_board._side_token
        next_hash = next_board.zobrist ^ side_token[self.next_player.other]

        # We can share the same set object safely – it only ever *grows*.
        self.hash_history.add(next_hash)

        return GameState(next_board,
                         self.next_player.other,
                         self, move,
                         next_hash,
                         self.hash_history)

    # ── game end & legality helpers ────────────────────────────
    def is_over(self) -> bool:
        if self.last_move is None:
            return False
        second_last = (self.previous_state.last_move
                        if self.previous_state else None)
        return (self.last_move.is_resign or
                (self.last_move.is_pass and second_last and second_last.is_pass))

    def does_move_violate_ko(self, player: Player, move: Move) -> bool:
        if not move.is_play:
            return False
        table, token = self.board._zobrist_table, self.board._side_token
        next_hash = (self.board.zobrist ^
                     table[(move.point.row, move.point.col, player)] ^
                     token[player.other])
        return next_hash in self.hash_history

    def _would_be_self_capture(self, player: Player, point: Point) -> bool:
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

    def is_valid_move(self, move: Move) -> bool:
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        if not self.board.is_on_grid(move.point):
            return False
        if self.board.get(move.point) is not None:
            return False
        if self._would_be_self_capture(self.next_player, move.point):
            return False
        if self.does_move_violate_ko(self.next_player, move):
            return False
        return True

    # ── shortcuts for agents ───────────────────────────────────
    def legal_moves(self) -> List[Move]:
        moves: List[Move] = []
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

    def winner(self) -> Optional[Player]:
        if not self.is_over():
            return None
        black, white = 0, 0
        for string in self.board._grid.values():
            if string.color == Player.black:
                black += len(string.stones)
            else:
                white += len(string.stones)
        return Player.black if black > white else Player.white
