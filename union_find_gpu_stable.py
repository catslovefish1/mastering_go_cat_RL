import torch, pandas as pd, numpy as np

# ───────────── Config ─────────────
NROWS, NCOLS = 4, 4                      # set 19,19 for a real board
N2           = NROWS * NCOLS
dtype        = torch.int16               # parent/colour storage

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

torch.manual_seed(0)

# ─────────── Build Union–Find table (7 cols) ───────────
uf = torch.empty((N2, 7), dtype=dtype, device=device)
flat = torch.arange(N2, dtype=dtype, device=device)

uf[:, 0] = flat                  # flat index   (immutable) – optional
uf[:, 1] = flat // NCOLS         # row
uf[:, 2] = flat % NCOLS          # col
uf[:, 3] = flat                  # parent = self
uf[:, 4] = -1                    # colour (–1 empty, 0 black, 1 white)
uf[:, 5] = 0                     # rank  (0 = single node)
uf[:, 6] = -1                    # spare / liberty count, unused for now

rows, cols   = uf[:, 1], uf[:, 2]
parent, colour = uf[:, 3], uf[:, 4]
rank            = uf[:, 5]

# ─────────── Constant neighbour matrix with −1 padding ───────────
OFF = torch.tensor([-NCOLS, NCOLS, -1, 1], dtype=torch.int32, device=device)
nbrs = flat.to(torch.int32).unsqueeze(1) + OFF                # (N²,4)

valid = (nbrs >= 0) & (nbrs < N2)
left_edge  = (cols == 0).unsqueeze(1)
right_edge = (cols == NCOLS-1).unsqueeze(1)
valid[:, 2] &= ~left_edge.squeeze()       # block wrap-left
valid[:, 3] &= ~right_edge.squeeze()      # block wrap-right

NEIGH_IDX = torch.where(valid, nbrs, torch.full_like(nbrs, -1))  # (N²,4)

# ─────────── Vectorised liberty calculator ───────────
def compute_liberties() -> torch.Tensor:
    """Return (N²,4) int16 tensor: liberty index or −1 per neighbour slot."""
    out  = torch.full_like(NEIGH_IDX, -1, dtype=dtype)
    mask = NEIGH_IDX != -1                       # real neighbours
    neigh_col = torch.zeros_like(out)
    neigh_col[mask] = colour[NEIGH_IDX[mask]]
    lib_mask = mask & (neigh_col == -1)          # empty neighbours
    out[lib_mask] = NEIGH_IDX[lib_mask].to(dtype)
    return out                                   # (N²,4)

# ─────────── GPU-only chain-liberty aggregation ───────────
def group_liberties_gpu():
    """
    Returns three device tensors:
        roots_unique     (U,) int32    – one root id per chain
        liberties_concat (K,) int16    – all liberties packed together
        counts           (U,) int32    – #liberties for each root
    Uses only ops available on MPS / CUDA / CPU.
    """
    # 1. current root of each point
    idx   = torch.arange(N2, device=device, dtype=torch.int32)
    roots = uf_find_vec(parent, idx)                 # (N²,)

    # 2. flatten (root, liberty) for every real liberty slot
    libs        = compute_liberties()                # (N²,4) int16
    roots_flat  = roots.unsqueeze(1).expand(-1, 4).reshape(-1)   # (N²*4,)
    libs_flat   = libs.reshape(-1)                                # (N²*4,)
    keep        = libs_flat != -1
    roots_kept  = roots_flat[keep]                                 # (M,)
    libs_kept   = libs_flat[keep].to(torch.int32)                  # (M,)

    # 3. make 64-bit radix key  root*(N²+1) + liberty
    key64 = roots_kept.to(torch.int64) * (N2 + 1) + libs_kept

    uniq = torch.unique(key64)             # (K,)  available on every backend
    uniq, _ = torch.sort(uniq)             # ensure root-major order

    # 4. decode keys
    roots_sorted     = (uniq // (N2 + 1)).to(torch.int32)  # (K,)
    liberties_sorted = (uniq %  (N2 + 1)).to(dtype)        # (K,) int16

    # 5. counts per root (no unique_dim)
    change  = torch.ones_like(roots_sorted, dtype=torch.bool)
    change[1:] = roots_sorted[1:] != roots_sorted[:-1]
    roots_unique = roots_sorted[change]                    # (U,)
    start_idx = torch.nonzero(change, as_tuple=False).flatten()
    start_idx = torch.cat([start_idx,
                           torch.tensor([roots_sorted.numel()],
                                        device=device)])
    counts = torch.diff(start_idx).to(torch.int32)         # (U,)

    return roots_unique, liberties_sorted, counts

# ─────────── Union–Find: find + union-by-rank ───────────
def uf_find_vec(parent: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx32 = idx.to(torch.int32)
    res32 = idx32.clone()
    for _ in range(16):                                    # sufficient for 19×19
        par16 = parent[res32]
        mask  = par16 != res32.to(dtype)
        if not mask.any(): break
        res32[mask]         = par16[mask].to(res32.dtype)
        parent[idx32[mask]] = res32[mask].to(dtype)        # path compression
    return res32                                           # (same shape as idx)

def uf_union_bulk(parent: torch.Tensor, pairs: torch.Tensor):
    if pairs.numel() == 0: return
    while True:
        ra = uf_find_vec(parent, pairs[:,0])
        rb = uf_find_vec(parent, pairs[:,1])
        diff = ra != rb
        if not diff.any(): break

        # union-by-rank
        a, b = ra[diff], rb[diff]                          # candidate roots
        swap = rank[a] < rank[b]                           # attach lower to higher
        a[swap], b[swap] = b[swap], a[swap]

        parent[a] = b.to(dtype)
        rank_eq = rank[a] == rank[b]
        rank[b[rank_eq]] += 1

# ─────────── Merge exactly ONE stone ───────────
def merge_one_stone(pos: int):
    c_self = colour[pos]
    for off in (-NCOLS, NCOLS, -1, 1):
        nbr = pos + off
        if nbr < 0 or nbr >= N2: continue
        if off == -1 and cols[pos] == 0: continue          # wrap-left
        if off ==  1 and cols[pos] == NCOLS-1: continue    # wrap-right
        if colour[nbr] == c_self:
            uf_union_bulk(parent, torch.tensor([[pos, nbr]],
                          dtype=torch.int32, device=device))

def add_stone(rc, col):
    pos = rc[0] * NCOLS + rc[1]
    colour[pos] = torch.tensor(col, dtype=dtype, device=device)
    merge_one_stone(pos)

# ─────────── Pretty-print helpers ───────────
def print_liberty_rows(title):
    libs = compute_liberties().cpu().tolist()
    lines = [f"{i:2d}: {tuple(libs[i])}" for i in range(N2)]
    print(f"\n{title}\n" + "\n".join(lines))

def print_chain_liberties():
    """
    Show every chain root, even those with 0 liberties, as
        root xx: [...]
    """
    # chains that actually own ≥1 liberty
    roots_u, lib_cat, cnts = group_liberties_gpu()
    roots_u_cpu  = roots_u.cpu()
    cnts_cpu     = cnts.cpu()
    lib_cat_cpu  = lib_cat.cpu()

    # all chains that exist on the board
    idx = torch.arange(N2, device=device, dtype=torch.int32)
    roots_all = torch.unique(uf_find_vec(parent, idx)).cpu()

    # make a quick lookup {root: count}
    cnt_lookup = {int(r): int(c) for r, c in zip(roots_u_cpu, cnts_cpu)}

    print("\nCHAIN-LEVEL LIBERTIES (including 0-liberty chains)")
    offset = 0
    for root in roots_all.tolist():
        n = cnt_lookup.get(root, 0)          # 0 if chain had no liberties
        libs = lib_cat_cpu[offset:offset+n].tolist() if n else []
        print(f"root {root:2d}: {libs}")
        if n:
            offset += n


# ─────────── Demo ───────────
def demo():
    for rc in [(0,1),(1,0),(1,2),(1,3),(2,1)]:
        add_stone(rc, 0)
    print_liberty_rows("LIBERTIES AFTER BOOTSTRAP")

    add_stone((1,1), 0)
    print_liberty_rows("LIBERTIES AFTER (1,1) PLAYED")
    print_chain_liberties()

if __name__ == "__main__":
    demo()
