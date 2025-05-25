# gotypes.py
# ------------------------------------------------------------------
import enum
from collections import namedtuple


class Player(enum.IntEnum):        # ← IntEnum gives value arithmetic “for free”
    black = 0
    white = 1

    @property
    def other(self) -> "Player":
        return Player.black if self is Player.white else Player.white


class Point(namedtuple("Point", "row col")):
    """Immutable (row, col) coordinate, 1-based."""
    __slots__ = ()

    # four cardinal neighbours
    def neighbors(self):
        r, c = self.row, self.col
        return [Point(r - 1, c), Point(r + 1, c), Point(r, c - 1), Point(r, c + 1)]

    # deepcopy is a no-op (Point is immutable)
    def __deepcopy__(self, memo):
        return self
