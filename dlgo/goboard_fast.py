# ko-safe Go engine – Zobrist + live (2,H,W) tensor on MPS/CPU
from __future__ import annotations
import os, secrets
import torch
from dlgo.gotypes import Player, Point

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MAX_BOARD = 19
_Z_KEYS = [[[secrets.randbits(64) for _ in range(2)]
            for _ in range(MAX_BOARD + 1)]
            for _ in range(MAX_BOARD + 1)]
_TURN_KEY = {Player.black: secrets.randbits(64),
             Player.white: secrets.randbits(64)}

# ── Move ────────────────────────────────────────────────────────────
class Move:
    def __init__(self, point: Point | None = None, *, is_pass=False):
        assert (point is not None) ^ is_pass
        self.point, self.is_play, self.is_pass = point, point is not None, is_pass
    @classmethod
    def play(cls, p):   return Move(point=p)
    @classmethod
    def pass_turn(cls): return Move(is_pass=True)

# ── GoString ────────────────────────────────────────────────────────
class GoString:
    def __init__(self, colour, stones, libs):
        self.color, self.stones, self.liberties = colour, set(stones), set(libs)
    def remove_liberty(self, p): self.liberties.discard(p)
    def add_liberty(self, p):    self.liberties.add(p)
    def merged_with(self, other):
        assert other.color == self.color
        return GoString(self.color,
                        self.stones | other.stones,
                        (self.liberties | other.liberties) - (self.stones | other.stones))
    @property
    def num_liberties(self): return len(self.liberties)
    def __eq__(self, o): return isinstance(o, GoString) and \
                          self.color==o.color and self.stones==o.stones and self.liberties==o.liberties
    def __hash__(self): return hash((self.color, frozenset(self.stones), frozenset(self.liberties)))

# ── Board ───────────────────────────────────────────────────────────
class Board:
    def __init__(self, rows, cols):
        self.num_rows, self.num_cols = rows, cols
        self._grid: dict[Point, GoString] = {}
        self._hash = 0
        self._tensor = torch.zeros((2, rows, cols), dtype=torch.uint8, device=DEVICE)

    def is_on_grid(self, p): return 1 <= p.row <= self.num_rows and 1 <= p.col <= self.num_cols
    def get(self, p):        return self._grid.get(p).color if p in self._grid else None
    def get_go_string(self, p): return self._grid.get(p)

    def copy(self):                                  # ← patched
        new = Board(self.num_rows, self.num_cols)
        new._grid = self._grid.copy()
        new._hash = self._hash
        new._tensor = self._tensor.clone().detach()  # no autograd compile
        return new

    def place_stone(self, player, point):
        assert self.is_on_grid(point) and self._grid.get(point) is None
        adj_same, adj_opp, libs = [], [], []
        for n in point.neighbors():
            if not self.is_on_grid(n): continue
            s = self._grid.get(n)
            if s is None: libs.append(n)
            elif s.color == player and s not in adj_same: adj_same.append(s)
            elif s.color != player and s not in adj_opp: adj_opp.append(s)

        self._hash ^= _Z_KEYS[point.row][point.col][player.value]
        self._tensor[player.value, point.row-1, point.col-1] = 1

        new_str = GoString(player, [point], libs)
        for s in adj_same: new_str = new_str.merged_with(s)
        for pnt in new_str.stones: self._grid[pnt] = new_str

        for other in adj_opp:
            other.remove_liberty(point)
        for other in adj_opp:
            if other.num_liberties == 0: self._remove_string(other)

    def _remove_string(self, string):
        for pt in string.stones:
            for n in pt.neighbors():
                s = self._grid.get(n)
                if s and s is not string: s.add_liberty(pt)
            self._tensor[string.color.value, pt.row-1, pt.col-1] = 0
            self._hash ^= _Z_KEYS[pt.row][pt.col][string.color.value]
            del self._grid[pt]

    @property
    def zobrist(self): return self._hash
    @property
    def tensor(self):  return self._tensor  # (2,H,W) uint8

# ── GameState ───────────────────────────────────────────────────────
class GameState:
    def __init__(self, board, next_player, prev, move, pos_hash):
        self.board, self.next_player = board, next_player
        self.previous_state, self.last_move = prev, move
        self.pos_hash = pos_hash

    def apply_move(self, move):
        nb = self.board.copy()
        if move.is_play: nb.place_stone(self.next_player, move.point)
        nh = nb.zobrist ^ _TURN_KEY[self.next_player.other]
        return GameState(nb, self.next_player.other, self, move, nh)

    @classmethod
    def new_game(cls, size):
        if isinstance(size, int): size = (size, size)
        b = Board(*size)
        return GameState(b, Player.black, None, None, b.zobrist ^ _TURN_KEY[Player.black])

    def is_over(self):
        if self.last_move is None: return False
        prev = self.previous_state.last_move if self.previous_state else None
        return self.last_move.is_pass and prev and prev.is_pass

    def _hash_after(self, pl, pt):
        b = self.board.copy(); b.place_stone(pl, pt)
        return b.zobrist ^ _TURN_KEY[pl.other]

    def does_move_violate_ko(self, pl, mv):
        if not mv.is_play: return False
        h = self._hash_after(pl, mv.point)
        st = self.previous_state
        while st:
            if st.pos_hash == h: return True
            st = st.previous_state
        return False

    def is_valid_move(self, mv):
        if self.is_over(): return False
        if mv.is_pass: return True
        if not self.board.is_on_grid(mv.point): return False
        if self.board.get(mv.point) is not None: return False
        b = self.board.copy(); b.place_stone(self.next_player, mv.point)
        if b.get_go_string(mv.point).num_liberties == 0: return False
        return not self.does_move_violate_ko(self.next_player, mv)

    def legal_moves(self):
        moves = []
        for r in range(1, self.board.num_rows+1):
            for c in range(1, self.board.num_cols+1):
                p = Point(r, c)
                if self.board.get(p) is None:
                    m = Move.play(p)
                    if self.is_valid_move(m): moves.append(m)
        moves.append(Move.pass_turn())
        return moves

    # tensor helper ---------------------------------------------------
    def as_tensor(self): return self.board.tensor  # (2,H,W) uint8

    def final_result(self):
        if not self.is_over(): return None
        b = sum(1 for p in self.board._grid if self.board.get(p) == Player.black)
        w = len(self.board._grid) - b
        return (Player.black if b>w else Player.white if w>b else None), b, w
