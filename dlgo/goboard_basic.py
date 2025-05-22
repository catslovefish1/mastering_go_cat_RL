from dlgo.gotypes import Player, Point


class Board:
    """
    Bare-bones Go board:
    * Keeps “who occupies what” (no liberties, capture, or ko yet)
    * Only validates that a play is on-board & on an empty point
    """
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid    = {}           # maps Point → Player

    # ----- helpers -------------------------------------------------------------
    def is_on_grid(self, point: Point) -> bool:
        """True if point lies inside the board edges."""
        return 1 <= point.row <= self.num_rows and \
               1 <= point.col <= self.num_cols

    def get(self, point: Point):
        """Return Player at point, or None if empty."""
        return self._grid.get(point)

    # ----- public action -------------------------------------------------------
    def place_stone(self, player: Player, point: Point):
        """Put stone down (no legality checks beyond ‘empty & on board’)."""
        if not self.is_on_grid(point):
            raise ValueError("Point off board")
        if point in self._grid:
            raise ValueError("Point already occupied")
        self._grid[point] = player