# -*- coding: utf-8 -*-
"""
GPU‑accelerated Go board core — **vectorised legality check**
─────────────────────────────────────────────────────────────
Adds `legal_mask(coords, col)` that evaluates move legality for an arbitrary
batch of candidate points *without mutating the board*.  The test is exact for
all standard Japanese / Tromp–Taylor rules (no‐suicide) because it implements
capture detection and friendly‑merge liberty union conservatively.

Shapes are annotated inline; this file supersedes the earlier versions.
"""

import torch

# ───────────── Config ─────────────
NROWS, NCOLS = 19, 19                 # (scalar, scalar)
N2           = NROWS * NCOLS          # scalar, 361 on 19×19
DTYPE        = torch.int16            # dtype alias

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

torch.manual_seed(0)

# ─────────── Union–Find table (7 cols) ───────────
uf   = torch.empty((N2, 7), dtype=DTYPE, device=device)        # (N², 7)
flat = torch.arange(N2, dtype=DTYPE, device=device)            # (N²,)

uf[:, 0] = flat                        # flat index               (N²,)
uf[:, 1] = flat // NCOLS               # row index                (N²,)
uf[:, 2] = flat %  NCOLS               # col index                (N²,)
uf[:, 3] = flat                        # parent                   (N²,)
uf[:, 4] = -1                          # colour −1/0/1            (N²,)
uf[:, 5] = 0                           # rank (unused)            (N²,)
uf[:, 6] = -1                          # spare / ko marker        (N²,)

row_idx, col_idx = uf[:, 1], uf[:, 2]                             # each (N²,)
parent,  colour  = uf[:, 3], uf[:, 4]                             # each (N²,)

# ─────────── Neighbour matrix (constant) ───────────
OFF  = torch.tensor([-NCOLS, NCOLS, -1, 1], dtype=torch.int32,
                    device=device)                              # (4,)
nbrs = flat.to(torch.int32).unsqueeze(1) + OFF                  # (N², 4)

valid = (0 <= nbrs) & (nbrs < N2)                               # (N², 4) bool
valid[:, 2] &= col_idx != 0
valid[:, 3] &= col_idx != NCOLS - 1
NEIGH_IDX = torch.where(valid, nbrs, torch.full_like(nbrs, -1)) # (N², 4)

# ─────────── Liberty map (N²,4) ───────────

def compute_liberties() -> torch.Tensor:
    out = torch.full_like(NEIGH_IDX, -1, dtype=DTYPE)           # (N², 4)
    m   = NEIGH_IDX != -1                                       # (N², 4) bool
    neigh_col          = torch.zeros_like(out)                  # (N², 4)
    neigh_col[m]       = colour[NEIGH_IDX[m]]                   # gather colours
    out[m & (neigh_col == -1)] = NEIGH_IDX[m & (neigh_col == -1)].to(DTYPE)
    return out

# ─────────── Chain‑liberty aggregation ───────────

def group_liberties_gpu():
    idx   = torch.arange(N2, device=device, dtype=torch.int32)  # (N²,)
    roots = uf_find_vec(parent, idx)                            # (N²,)
    libs        = compute_liberties()                           # (N², 4)
    roots_flat  = roots.unsqueeze(1).expand(-1, 4).reshape(-1)  # (N²·4,)
    libs_flat   = libs.reshape(-1)                              # (N²·4,)
    keep        = libs_flat != -1                               # (N²·4,) bool
    roots_kept  = roots_flat[keep]                              # (K,)
    libs_kept   = libs_flat[keep].to(torch.int32)               # (K,)

    key64 = roots_kept.to(torch.int64) * (N2 + 1) + libs_kept   # (K,) unique key
    uniq  = torch.sort(torch.unique(key64)).values              # (K,)

    roots_sorted     = (uniq // (N2 + 1)).to(torch.int32)       # (K,)
    change           = torch.ones_like(roots_sorted, dtype=torch.bool)
    change[1:]       = roots_sorted[1:] != roots_sorted[:-1]
    roots_unique     = roots_sorted[change]                     # (U,)

    start  = torch.nonzero(change, as_tuple=False).flatten()    # (U,)
    start  = torch.cat([start,
                        torch.tensor([roots_sorted.numel()], device=device)])
    counts = torch.diff(start).to(torch.int32)                  # (U,)
    return roots_unique, counts                                 # (U,), (U,)

# Pre‑compute per‑point root and per‑root liberty counts
# Updated lazily by caller after board changes.
POINT_ROOTS   = torch.arange(N2, device=device, dtype=torch.int32)  # (N²,) init
ROOT_LIBS     = torch.zeros(N2, device=device, dtype=torch.int32)   # (N²,) counts


def _refresh_root_and_libs():
    """Recompute POINT_ROOTS and ROOT_LIBS from global board state."""
    global POINT_ROOTS, ROOT_LIBS
    POINT_ROOTS = uf_find_vec(parent, torch.arange(N2, device=device,
                                                  dtype=torch.int32))  # (N²,)
    ROOT_LIBS.zero_()
    roots, cnts = group_liberties_gpu()
    ROOT_LIBS[roots] = cnts

# ─────────── Union–Find primitives ───────────

def uf_find_vec(par: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx32 = idx.to(torch.int32)
    res32 = idx32.clone()
    for _ in range(16):
        p = par[res32]
        m = p != res32.to(DTYPE)
        if not torch.any(m):
            break
        res32[m]      = p[m].to(res32.dtype)
        par[idx32[m]] = res32[m].to(DTYPE)
    return res32

def uf_union_bulk(par: torch.Tensor, pairs: torch.Tensor):
    if pairs.numel() == 0:
        return
    while True:
        ra = uf_find_vec(par, pairs[:, 0])
        rb = uf_find_vec(par, pairs[:, 1])
        diff = ra != rb
        if not torch.any(diff):
            break
        hi = torch.maximum(ra[diff], rb[diff])
        lo = torch.minimum(ra[diff], rb[diff])
        par[hi] = lo.to(DTYPE)

# ─────────── Board‑update helpers ───────────

def merge_one_stone(pos: int):
    col_self = colour[pos]
    for off in (-NCOLS, NCOLS, -1, 1):
        nbr = pos + off
        if nbr < 0 or nbr >= N2:
            continue
        if off == -1 and col_idx[pos] == 0:
            continue
        if off == 1 and col_idx[pos] == NCOLS - 1:
            continue
        if colour[nbr] == col_self:
            pair = torch.tensor([[pos, nbr]], dtype=torch.int32, device=device)
            uf_union_bulk(parent, pair)

def add_stone(coord, col):
    pos = coord if isinstance(coord, int) else coord[0] * NCOLS + coord[1]
    colour[pos] = torch.tensor(col, dtype=DTYPE, device=device)
    merge_one_stone(pos)
    return pos

# ─────────── Vectorised legality (no side‑effects) ───────────

def legal_mask(coords: torch.Tensor, col: int) -> torch.Tensor:
    """Return a Bool tensor `legal` with same length as `coords` (1‑D int32).

    The board itself is *not* modified.  Caller must run `_refresh_root_and_libs()`
    if stones have been added/removed since the last call.
    """
    coords = coords.to(torch.int32).flatten()                    # (M,)
    M = coords.numel()

    # Pre‑slice frequently used tensors
    neigh4 = NEIGH_IDX[coords]                                   # (M, 4)

    # 1. Occupancy test
    empty_mask = colour[coords] == -1                            # (M,) bool

    # 2. Immediate empty neighbour → always legal
    neigh_is_valid = neigh4 != -1                                # (M, 4) bool
    neigh_empty    = torch.zeros_like(neigh4, dtype=torch.bool)  # (M, 4)
    neigh_empty[neigh_is_valid] = colour[neigh4[neigh_is_valid]] == -1
    has_liberty_now = torch.any(neigh_empty, dim=1)              # (M,) bool

    # 3. Capture test: opponent chains with single liberty (the move point)
    opp              = 1 - col
    neigh_opp_mask   = neigh_is_valid & (colour[neigh4] == opp)  # (M, 4) bool
    opp_roots        = torch.where(neigh_opp_mask,
                                   POINT_ROOTS[neigh4],
                                   torch.full_like(neigh4, -1))  # (M, 4)
    # liberty count == 1 ?
    opp_root_libs    = torch.zeros_like(opp_roots)
    valid_opp        = opp_roots != -1
    opp_root_libs[valid_opp] = ROOT_LIBS[opp_roots[valid_opp]]
    capture_possible = torch.any((opp_root_libs == 1) & valid_opp, dim=1)  # (M,)

    # 4. Friendly merge test: after placing, does merged chain still have libs?
    neigh_self_mask  = neigh_is_valid & (colour[neigh4] == col)  # (M, 4) bool
    self_roots       = torch.where(neigh_self_mask,
                                   POINT_ROOTS[neigh4],
                                   torch.full_like(neigh4, -1))  # (M, 4)
    # For each candidate, compute max(lib_count[root] - 1) across its own roots.
    self_root_libs   = torch.zeros_like(self_roots)
    valid_self       = self_roots != -1
    self_root_libs[valid_self] = ROOT_LIBS[self_roots[valid_self]] - 1  # subtract move
    still_libs_after = torch.any(self_root_libs > 0, dim=1)             # (M,)

    legal = empty_mask & (has_liberty_now | capture_possible | still_libs_after)
    return legal

# ─────────── Play‑move (unchanged except refresh call) ───────────

def play_move(coord, col) -> bool:
    pos = coord if isinstance(coord, int) else coord[0] * NCOLS + coord[1]
    if colour[pos] != -1:
        return False
    colour_backup = colour.clone(); parent_backup = parent.clone()
    add_stone(coord, col)

    _refresh_root_and_libs()                                # update caches

    root_new = POINT_ROOTS[pos]
    libs_new = ROOT_LIBS[root_new]

    opp       = 1 - col
    nbr4      = NEIGH_IDX[pos]
    opp_mask  = (nbr4 != -1) & (colour[nbr4] == opp)
    captured_roots = torch.tensor([], dtype=torch.int32, device=device)
    if torch.any(opp_mask):
        nbr_roots = POINT_ROOTS[nbr4[opp_mask]]
        uniq      = torch.unique(nbr_roots)
        dead_mask = ROOT_LIBS[uniq] == 0
        captured_roots = uniq[dead_mask]

    legal = (libs_new > 0) | (captured_roots.numel() > 0)
    if not bool(legal):
        colour.copy_(colour_backup); parent.copy_(parent_backup)
        _refresh_root_and_libs()
        return False

    if captured_roots.numel() > 0:
        all_pts   = torch.arange(N2, device=device, dtype=torch.int32)
        reps      = POINT_ROOTS
        kill_mask = torch.isin(reps, captured_roots)
        colour[kill_mask] = -1
        parent[kill_mask] = all_pts[kill_mask].to(DTYPE)
        _refresh_root_and_libs()

    return True

# ─────────── Pretty‑print helpers (omitted for brevity) ───────────

# ─────────── Demo ───────────

def demo_initialized_board():
    """Demo: initialize board with specific stones and return legal moves."""
    global NROWS, NCOLS, N2, uf, flat, row_idx, col_idx, parent, colour
    global NEIGH_IDX, POINT_ROOTS, ROOT_LIBS

    # Reset to 4×4 board
    NROWS, NCOLS, N2 = 4, 4, 16
    uf   = torch.empty((N2, 7), dtype=DTYPE, device=device)
    flat = torch.arange(N2, dtype=DTYPE, device=device)
    uf[:, 0] = flat
    uf[:, 1] = flat // NCOLS  # row
    uf[:, 2] = flat % NCOLS   # col
    uf[:, 3] = flat           # parent
    uf[:, 4] = -1            # colour
    uf[:, 5] = 0             # rank
    uf[:, 6] = -1            # spare
    
    row_idx, col_idx = uf[:, 1], uf[:, 2]
    parent, colour   = uf[:, 3], uf[:, 4]

    # Rebuild neighbor index for 4×4
    OFF  = torch.tensor([-NCOLS, NCOLS, -1, 1], dtype=torch.int32, device=device)
    nbrs = flat.to(torch.int32).unsqueeze(1) + OFF
    valid = (0 <= nbrs) & (nbrs < N2)
    valid[:, 2] &= col_idx != 0
    valid[:, 3] &= col_idx != NCOLS - 1
    NEIGH_IDX = torch.where(valid, nbrs, torch.full_like(nbrs, -1))

    # Initialize board with black and white stones
    black_stones = [(0, 1), (1, 0), (1, 2), (1, 3), (2, 1)]
    white_stones = [(0, 2), (2, 2), (3, 1)]  # Add some white stones
    
    print("Placing black stones at:", black_stones)
    for rc in black_stones:
        add_stone(rc, 0)  # 0 = black
        
    print("Placing white stones at:", white_stones)
    for rc in white_stones:
        add_stone(rc, 1)  # 1 = white
    
    # Must refresh caches after board changes
    _refresh_root_and_libs()

    # Show current board
    print("\nCurrent board (● = black, ○ = white, . = empty):")
    board_display = colour.clone().reshape(NROWS, NCOLS)
    symbols = {-1: ".", 0: "●", 1: "○"}
    for r in range(NROWS):
        row = " ".join(symbols[int(c)] for c in board_display[r])
        print(f"  {row}")
    
    # Get legal moves for both colors
    all_positions = torch.arange(N2, device=device, dtype=torch.int32)
    white_legal_mask = legal_mask(all_positions, col=1)  # White
    black_legal_mask = legal_mask(all_positions, col=0)  # Black
    
    # Convert to coordinate lists
    white_legal_moves = [(i // NCOLS, i % NCOLS) for i in range(N2) if white_legal_mask[i]]
    black_legal_moves = [(i // NCOLS, i % NCOLS) for i in range(N2) if black_legal_mask[i]]
    
    print(f"\nLegal moves for White: {white_legal_moves}")
    print(f"Legal moves for Black: {black_legal_moves}")
    
    # Visualize legal moves
    print("\nLegal moves visualization:")
    print("White (◦ = legal):")
    for r in range(NROWS):
        row_str = ""
        for c in range(NCOLS):
            pos = r * NCOLS + c
            if colour[pos] == 0:
                row_str += "● "
            elif colour[pos] == 1:
                row_str += "○ "
            elif white_legal_mask[pos]:
                row_str += "◦ "
            else:
                row_str += "· "
        print(f"  {row_str}")
    
    print("\nBlack (✓ = legal):")
    for r in range(NROWS):
        row_str = ""
        for c in range(NCOLS):
            pos = r * NCOLS + c
            if colour[pos] == 0:
                row_str += "● "
            elif colour[pos] == 1:
                row_str += "○ "
            elif black_legal_mask[pos]:
                row_str += "✓ "
            else:
                row_str += "· "
        print(f"  {row_str}")
    
    return {
        'white_legal': white_legal_moves,
        'black_legal': black_legal_moves,
        'board_state': board_display.cpu().numpy()
    }

# Replace the original demo with this one
demo_batch_legality = demo_initialized_board

if __name__ == "__main__":
    # Run the demo
    result = demo_initialized_board()
    
    # Example: try playing at (1,1) which would connect the black stones
    print("\n--- Testing specific moves ---")
    test_moves = [(1, 1), (0, 0), (3, 3)]
    
    for move in test_moves:
        # Create a copy to test without modifying board
        coords_tensor = torch.tensor([move[0] * NCOLS + move[1]], 
                                   dtype=torch.int32, device=device)
        white_can_play = legal_mask(coords_tensor, col=1)[0]
        black_can_play = legal_mask(coords_tensor, col=0)[0]
        
        print(f"Position {move}: White={'✓' if white_can_play else '✗'}, "
              f"Black={'✓' if black_can_play else '✗'}")