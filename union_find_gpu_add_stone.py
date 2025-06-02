# -*- coding: utf-8 -*-
"""
GPU‑accelerated Go board core — *TorchScript‑free edition*  
**With explicit shape comments**
──────────────────────────────────────────────────────────
Each tensor creation or major assignment now carries an inline comment showing
its run‑time shape.  Shapes use `N² = NROWS × NCOLS`; thus on the default 19 × 19
board we have `N² = 361`.

(No behavioural changes compared with the previous commit.)
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
# broadcast add → vertical stack of 4 offsets per point
nbrs = flat.to(torch.int32).unsqueeze(1) + OFF                  # (N², 4)

valid = (0 <= nbrs) & (nbrs < N2)                               # (N², 4) bool
valid[:, 2] &= col_idx != 0
valid[:, 3] &= col_idx != NCOLS - 1
NEIGH_IDX = torch.where(valid, nbrs, torch.full_like(nbrs, -1)) # (N², 4)

# ─────────── Liberty map (N²,4) ───────────

def compute_liberties() -> torch.Tensor:
    """Return a (N²,4) matrix whose entries are neighbour indices that are *empty*."""
    out = torch.full_like(NEIGH_IDX, -1, dtype=DTYPE)           # (N², 4)
    m   = NEIGH_IDX != -1                                       # (N², 4) bool
    neigh_col          = torch.zeros_like(out)                  # (N², 4)
    neigh_col[m]       = colour[NEIGH_IDX[m]]                   # gather colours
    out[m & (neigh_col == -1)] = NEIGH_IDX[m & (neigh_col == -1)].to(DTYPE)
    return out

# ─────────── Chain‑liberty aggregation ───────────

def group_liberties_gpu():
    """Return `(roots_unique, liberty_cat, counts)`.
    * `roots_unique`  – (U,) int32     : chain representatives having ≥1 liberty
    * `liberty_cat`   – (K,) int16     : concatenated liberty coords
    * `counts`        – (U,) int32     : how many liberties per root
    where `sum(counts) == K`.
    """
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
    liberties_sorted = (uniq %  (N2 + 1)).to(DTYPE)             # (K,)

    change = torch.ones_like(roots_sorted, dtype=torch.bool)    # (K,) bool
    change[1:] = roots_sorted[1:] != roots_sorted[:-1]
    roots_unique = roots_sorted[change]                         # (U,)

    start  = torch.nonzero(change, as_tuple=False).flatten()    # (U,)
    start  = torch.cat([start,
                        torch.tensor([roots_sorted.numel()], device=device)])
    counts = torch.diff(start).to(torch.int32)                  # (U,)
    return roots_unique, liberties_sorted, counts

# ─────────── Union–Find primitives ───────────

def uf_find_vec(par: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Vectorised find with path halving; `idx` → same shape output."""
    idx32 = idx.to(torch.int32)         # (⋯,)
    res32 = idx32.clone()               # (⋯,)
    for _ in range(16):
        p = par[res32]                  # (⋯,) parent lookup
        m = p != res32.to(DTYPE)        # (⋯,) bool
        if not torch.any(m):
            break
        res32[m]      = p[m].to(res32.dtype)
        par[idx32[m]] = res32[m].to(DTYPE)
    return res32                        # (⋯,)

def uf_union_bulk(par: torch.Tensor, pairs: torch.Tensor):
    """`pairs` is (P,2) int32 of node indices to merge."""
    if pairs.numel() == 0:
        return
    while True:
        ra = uf_find_vec(par, pairs[:, 0])   # (P,)
        rb = uf_find_vec(par, pairs[:, 1])   # (P,)
        diff = ra != rb                      # (P,) bool
        if not torch.any(diff):
            break
        hi = torch.maximum(ra[diff], rb[diff])  # (≤P,)
        lo = torch.minimum(ra[diff], rb[diff])  # (≤P,)
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
            pair = torch.tensor([[pos, nbr]], dtype=torch.int32, device=device)  # (1,2)
            uf_union_bulk(parent, pair)

def add_stone(coord, col):
    """Place stone of `col` at `coord`; return flat index."""
    pos = coord if isinstance(coord, int) else coord[0] * NCOLS + coord[1]
    colour[pos] = torch.tensor(col, dtype=DTYPE, device=device)
    merge_one_stone(pos)
    return pos

# ─────────── NEW: Play‑move with legality & capture ───────────

def play_move(coord, col) -> bool:
    """Attempt to play **col** (0/1) at **coord**; return True on success."""
    pos = coord if isinstance(coord, int) else coord[0] * NCOLS + coord[1]
    if colour[pos] != -1:                       # occupied check
        return False

    colour_backup = colour.clone()              # (N²,) copy
    parent_backup = parent.clone()              # (N²,) copy

    add_stone(coord, col)
    root_new = uf_find_vec(parent,
                           torch.tensor([pos], device=device, dtype=torch.int32))[0]  # scalar tensor

    roots_u, _, cnts = group_liberties_gpu()    # roots_u:(U,), cnts:(U,)
    lib_counts = torch.zeros(N2, dtype=torch.int32, device=device)  # (N²,)
    lib_counts[roots_u] = cnts
    libs_new = lib_counts[root_new]             # scalar tensor

    opp       = 1 - col
    nbr4      = NEIGH_IDX[pos]                  # (4,)
    opp_mask  = (nbr4 != -1) & (colour[nbr4] == opp)  # (4,) bool

    captured_roots = torch.tensor([], dtype=torch.int32, device=device)
    if torch.any(opp_mask):
        nbr_roots = uf_find_vec(parent, nbr4[opp_mask])
        uniq      = torch.unique(nbr_roots)
        if uniq.numel() > 0:
            dead_mask = lib_counts[uniq] == 0
            captured_roots = uniq[dead_mask]

    legal = (libs_new > 0) | (captured_roots.numel() > 0)
    legal_flag = bool(legal.item())

    if not legal_flag:
        # Roll back
        colour.copy_(colour_backup)
        parent.copy_(parent_backup)
        return False

    # Remove captured stones (if any)
    if captured_roots.numel() > 0:
        all_pts   = torch.arange(N2, device=device, dtype=torch.int32)
        reps      = uf_find_vec(parent, all_pts)
        kill_mask = torch.isin(reps, captured_roots)
        colour[kill_mask] = -1
        parent[kill_mask] = all_pts[kill_mask].to(DTYPE)

    return True

# ─────────── Pretty‑print helpers (CPU‑side) ───────────

def print_liberty_rows(title):
    libs = compute_liberties().cpu().tolist()
    print(f"\n{title}")
    for i in range(N2):
        print(f"{i:2d}: {tuple(libs[i])}")


def print_chain_liberties():
    roots_u, lib_cat, cnts = group_liberties_gpu()
    lib_dict = {}
    off = 0
    for r, n in zip(roots_u.tolist(), cnts.tolist()):
        lib_dict[r] = lib_cat[off:off + n].cpu().tolist()
        off += n

    idx  = torch.arange(N2, device=device, dtype=torch.int32)
    reps = uf_find_vec(parent, idx).cpu().tolist()
    stones_present = {r for r, c in zip(reps, colour.cpu().tolist()) if c != -1}

    print("\nCHAIN LIBERTIES (all root ids 0…N²−1)")
    for root in range(N2):
        mark = "*" if root in stones_present else " "
        print(f"root {root:2d}{mark}: {lib_dict.get(root, [])}")

# ─────────── Demo ───────────

def demo():
    # Board setup (example on 4×4 for brevity)
    global NROWS, NCOLS, N2
    NROWS, NCOLS, N2 = 4, 4, 16
    # Reset tensors to the smaller board
    global uf, flat, row_idx, col_idx, parent, colour, NEIGH_IDX
    uf   = torch.empty((N2, 7), dtype=DTYPE, device=device)
    flat = torch.arange(N2, dtype=DTYPE, device=device)
    uf[:, 0] = flat
    uf[:, 1] = flat // NCOLS
    uf[:, 2] = flat %  NCOLS
    uf[:, 3] = flat
    uf[:, 4] = -1
    uf[:, 5] = 0
    uf[:, 6] = -1
    row_idx, col_idx = uf[:, 1], uf[:, 2]
    parent,  colour  = uf[:, 3], uf[:, 4]
    OFF  = torch.tensor([-NCOLS, NCOLS, -1, 1], dtype=torch.int32, device=device)
    nbrs = flat.to(torch.int32).unsqueeze(1) + OFF
    valid = (0 <= nbrs) & (nbrs < N2)
    valid[:, 2] &= col_idx != 0
    valid[:, 3] &= col_idx != NCOLS - 1
    NEIGH_IDX = torch.where(valid, nbrs, torch.full_like(nbrs, -1))

    current_board = [((0, 1), 0), ((1, 0), 0)]
    for rc, c in current_board:
        add_stone(rc, c)

    print_liberty_rows("LIBERTIES AFTER SETUP")

    if play_move((0, 0), 0):
        print("\n✔ legal move")
    else:
        print("\n✘ illegal (suicide)")

    print_liberty_rows("BOARD AFTER MOVE")
    print_chain_liberties()

if __name__ == "__main__":
    demo()
