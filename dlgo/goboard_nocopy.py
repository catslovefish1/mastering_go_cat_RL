# goboard_nocopy.py  – minimal Go engine, single-board apply/undo version
# (latest revision: fixes nested _apply assertion in is_valid_move)
from __future__ import annotations
import os, secrets
from collections import namedtuple
import torch
from dlgo.gotypes import Player, Point

# ── device setup ────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ── Zobrist constants (19 × 19 max) ─────────────────────────────────────────
MAX_BOARD = 19
_Z_KEYS = [[[secrets.randbits(64) for _ in range(2)]
            for _ in range(MAX_BOARD + 1)]
           for _ in range(MAX_BOARD + 1)]
_TURN_KEY = {Player.black: secrets.randbits(64),
             Player.white: secrets.randbits(64)}



# ── Delta: what _apply() needs to undo ─────────────────────────────────────
Delta = namedtuple('Delta',
                   ['placed', 'captured', 'merged_from',
                    'old_libs', 'hash_before'])

# ── basic data classes ─────────────────────────────────────────────────────
class Move:
    def __init__(self, point: Point | None = None, *, is_pass: bool = False):
        assert (point is not None) ^ is_pass
        self.point, self.is_pass = point, is_pass
        self.is_play = point is not None

    @classmethod
    def play(cls, p):       return Move(point=p)
    @classmethod
    def pass_turn(cls):     return Move(is_pass=True)

class GoString:
    def __init__(self, colour, stones, libs):
        self.color, self.stones, self.liberties = colour, set(stones), set(libs)
    def remove_liberty(self, p): self.liberties.discard(p)
    def add_liberty   (self, p): self.liberties.add(p)
    def merged_with(self, other):
        assert other.color == self.color
        return GoString(self.color,
                        self.stones | other.stones,
                        (self.liberties | other.liberties) -
                        (self.stones  | other.stones))
    @property
    def num_liberties(self): return len(self.liberties)

class Board:
    def __init__(self, rows, cols):
        self.num_rows, self.num_cols = rows, cols
        self._grid: dict[Point, GoString] = {}
        self._hash = 0
        self._tensor = torch.zeros((2, rows, cols), dtype=torch.uint8,
                                   device=DEVICE)

    # ---- fast helpers -----------------------------------------------------
    def is_on_grid(self, p): return 1 <= p.row <= self.num_rows and 1 <= p.col <= self.num_cols
    def get(self, p): return self._grid.get(p).color if p in self._grid else None
    def get_go_string(self, p): return self._grid.get(p)

    # ---- place a stone in-place, return Delta to undo ---------------------
    def _apply(self, player: Player, point: Point) -> Delta:
        assert self.is_on_grid(point) and self._grid.get(point) is None

        adj_same, adj_opp, libs = [], [], []
        for n in point.neighbors():
            if not self.is_on_grid(n): continue
            s = self._grid.get(n)
            if s is None:
                libs.append(n)
            elif s.color == player and s not in adj_same:
                adj_same.append(s)
            elif s and s.color != player and s not in adj_opp:
                adj_opp.append(s)

        old_libs   = {g: set(g.liberties) for g in adj_same + adj_opp}
        hash_before = self._hash

        # tensor + hash
        self._hash ^= _Z_KEYS[point.row][point.col][player.value]
        self._tensor[player.value, point.row-1, point.col-1] = 1

        new_str, merged_from = GoString(player, [point], libs), []
        for s in adj_same:
            merged_from.append(s)
            new_str = new_str.merged_with(s)
        for pt in new_str.stones:
            self._grid[pt] = new_str

        captured = []
        for other in adj_opp:
            other.remove_liberty(point)
        for other in adj_opp:
            if other.num_liberties == 0:
                captured.append(other)
                self._remove_string(other)

        return Delta(point, captured, merged_from, old_libs, hash_before)

    # ---- undo a Delta -----------------------------------------------------
    def _undo(self, d: Delta):
        # restore captures
        for string in d.captured:
            for pt in string.stones:
                self._grid[pt] = string
                self._tensor[string.color.value, pt.row-1, pt.col-1] = 1
                self._hash ^= _Z_KEYS[pt.row][pt.col][string.color.value]

        # remove provisional string & stone
        pt = d.placed
        cur_str = self._grid.get(pt)
        if cur_str:
            for s in cur_str.stones:
                self._grid.pop(s, None)
            self._tensor[cur_str.color.value, pt.row-1, pt.col-1] = 0
            self._hash ^= _Z_KEYS[pt.row][pt.col][cur_str.color.value]

        # restore merged groups
        for g, libs in d.old_libs.items():
            for s in g.stones:
                self._grid[s] = g
            g.liberties.clear(); g.liberties.update(libs)

        self._hash = d.hash_before

    # ---- internal capture helper -----------------------------------------
    def _remove_string(self, string: GoString):
        for pt in string.stones:
            for n in pt.neighbors():
                s = self._grid.get(n)
                if s and s is not string:
                    s.add_liberty(pt)
            self._tensor[string.color.value, pt.row-1, pt.col-1] = 0
            self._hash ^= _Z_KEYS[pt.row][pt.col][string.color.value]
            del self._grid[pt]

    # ---- read-only views --------------------------------------------------
    @property
    def zobrist(self): return self._hash
    @property
    def tensor (self): return self._tensor      # (2,H,W) uint8

# ── GameState (copy-free) ───────────────────────────────────────────────────
class GameState:
    def __init__(self, board, next_player, prev, move, pos_hash):
        self.board, self.next_player = board, next_player
        self.previous_state, self.last_move = prev, move
        self.pos_hash = pos_hash

    # permanent apply (for accepted moves)
    def apply_move(self, move: Move):
        if move.is_play:
            self.board._apply(self.next_player, move.point)
        nh = self.board.zobrist ^ _TURN_KEY[self.next_player.other]
        return GameState(self.board, self.next_player.other, self, move, nh)

    @classmethod
    def new_game(cls, size):
        size = (size, size) if isinstance(size, int) else size
        b = Board(*size)
        return GameState(b, Player.black, None, None,
                         b.zobrist ^ _TURN_KEY[Player.black])

    def is_over(self):
        if self.last_move is None: return False
        prev = self.previous_state.last_move if self.previous_state else None
        return self.last_move.is_pass and prev and prev.is_pass

    # ko helper -------------------------------------------------------------
    def _hash_after(self, pl, pt):
        d = self.board._apply(pl, pt)
        h = self.board.zobrist ^ _TURN_KEY[pl.other]
        self.board._undo(d)
        return h

    def does_move_violate_ko(self, pl, mv):
        if not mv.is_play: return False
        h = self._hash_after(pl, mv.point)
        st = self.previous_state
        while st:
            if st.pos_hash == h: return True
            st = st.previous_state
        return False

    # ---- validity (fixed order: ko first, then self-capture) --------------
    def is_valid_move(self, mv: Move):
        if self.is_over():                                return False
        if mv.is_pass:                                    return True
        if not self.board.is_on_grid(mv.point):           return False
        if self.board.get(mv.point) is not None:          return False
        if self.does_move_violate_ko(self.next_player, mv): return False

        d = self.board._apply(self.next_player, mv.point)
        suicide = (self.board.get_go_string(mv.point).num_liberties == 0)
        self.board._undo(d)

        return not suicide

    # ---- enumerate -------------------------------------------------------
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
        return moves

    # ---- misc helpers -----------------------------------------------------
    def as_tensor(self): return self.board.tensor
    def final_result(self):
        if not self.is_over(): return None
        b = sum(1 for p in self.board._grid if self.board.get(p) == Player.black)
        w = len(self.board._grid) - b
        winner = Player.black if b > w else Player.white if w > b else None
        return winner, b, w

# ── Random bot (unchanged) ─────────────────────────────────────────────────
class TensorRandomBot:
    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size
    def select_move(self, game_state):
        states = (game_state,) if self.batch_size is None else game_state
        legal_lists = []
        for st in states:
            mv = st.legal_moves()
            stones = [m for m in mv if m.is_play]
            legal_lists.append(stones or mv)
        max_len = max(len(lst) for lst in legal_lists)
        pad = -1
        idx = torch.full((len(states), max_len), pad,
                         dtype=torch.long, device=DEVICE)
        for r, lst in enumerate(legal_lists):
            idx[r, :len(lst)] = torch.arange(len(lst), device=DEVICE)
        probs = (idx != pad).float()
        choice = torch.multinomial(probs, 1).squeeze(1)
        chosen = [legal_lists[r][int(c)] for r, c in enumerate(choice)]
        return chosen[0] if self.batch_size is None else chosen
