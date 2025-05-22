# gotypes.py
import enum
from collections import namedtuple

class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self is Player.white else Player.white


class Point(namedtuple('Point', 'row col')):
    """Immutable (row, col) coordinate, 1-based."""
    __slots__ = ()                     # no per-instance dict → smaller & faster

    # NEW ↓↓↓
    def neighbors(self):
        """Return the four directly-adjacent points (up, down, left, right)."""
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

    # Optional but handy: make deepcopy a no-op (Point is already immutable)
    def __deepcopy__(self, memodict={}):
        return self
