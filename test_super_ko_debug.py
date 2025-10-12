#!/usr/bin/env python3
from __future__ import annotations
import torch
from engine.GoLegalMoveChecker import GoLegalMoveChecker
from engine.tensor_native_stable import TensorBoard, Stone

def xor_reduce_1d(vec: torch.Tensor) -> torch.Tensor:
    """Bitwise XOR-reduce a 1D tensor."""
    if vec.numel() == 0:
        return torch.zeros((), dtype=torch.long, device=vec.device)
    x = vec
    while x.numel() > 1:
        n = x.numel()
        x = torch.bitwise_xor(x[: n//2], x[n//2 : (n//2)*2])
        if n % 2:
            x = torch.bitwise_xor(x, x[-1:].expand_as(x))
    return x.squeeze(0) if x.dim() > 0 else x

def full_hash(tb: TensorBoard, board_2d: torch.Tensor) -> int:
    """Compute full zobrist hash from a board state."""
    H = W = tb.board_size
    idx = (board_2d.to(torch.long) + 1).view(-1)  # -1,0,1 -> 0,1,2
    lin = torch.arange(H * W, device=tb.device, dtype=torch.long)
    z = tb.Zpos[lin, idx]
    return int(xor_reduce_1d(z).item())

def build_tb(board_flat, color="black"):
    """Build a TensorBoard with given board state."""
    tb = TensorBoard(batch_size=1, board_size=3, enable_super_ko=True, enable_timing=False)
    H = W = 3
    b = torch.tensor(board_flat, dtype=torch.int8, device=tb.device).view(H, W)
    tb.board[0] = b
    tb.current_player[0] = 0 if color == "black" else 1
    tb.current_hash[0] = full_hash(tb, b)
    tb._invalidate_cache()
    return tb

def compute_hash_delta(tb, r, c, cap_mask):
    """Manually compute the hash delta for placing a stone."""
    H = W = tb.board_size
    lin = r * W + c
    player = int(tb.current_player[0].item())
    
    # Placement delta: EMPTY -> player
    Z = tb.Zpos
    place_old = Z[lin, 0]  # empty
    place_new = Z[lin, player + 1]  # black=1, white=2
    place_delta = place_old ^ place_new
    
    # Capture delta
    opp = 1 - player
    cap_flat = cap_mask.view(-1)
    cap_indices = cap_flat.nonzero(as_tuple=True)[0]
    
    cap_delta = torch.zeros((), dtype=torch.long, device=tb.device)
    for idx in cap_indices:
        cap_old = Z[idx, opp + 1]  # opponent stone
        cap_new = Z[idx, 0]  # empty
        cap_delta ^= (cap_old ^ cap_new)
    
    # Total delta
    total_delta = place_delta ^ cap_delta
    new_hash = tb.current_hash[0] ^ total_delta
    
    print(f"  Manual delta calculation:")
    print(f"    place_old (empty at {lin}): {int(place_old.item())}")
    print(f"    place_new ({['black','white'][player]} at {lin}): {int(place_new.item())}")
    print(f"    place_delta: {int(place_delta.item())}")
    print(f"    captured positions: {cap_indices.tolist()}")
    print(f"    cap_delta: {int(cap_delta.item())}")
    print(f"    total_delta: {int(total_delta.item())}")
    print(f"    new_hash (current ^ delta): {int(new_hash.item())}")
    
    return int(new_hash.item())

def main():
    print("=== SUPER-KO DEBUG ===\n")
    
    # Initial board state (move 31 POST)
    pre_flat = [
        0,  0, -1,
        1,  0,  1,
       -1,  1,  1,
    ]
    r, c = 2, 0  # Black wants to play here
    
    print("Initial board (Black to play at (2,0)):")
    for i in range(3):
        row = pre_flat[i*3:(i+1)*3]
        print("  ", [' B' if x == 0 else ' W' if x == 1 else ' .' for x in row])
    print()
    
    # Build initial position
    tb = build_tb(pre_flat, "black")
    pre_hash = int(tb.current_hash[0].item())
    
    # Get capture info
    legal0 = tb.legal_moves()
    cap = tb._last_capture_info
    roots = cap["roots"]
    caps = cap["capture_groups"]
    
    H = W = tb.board_size
    lin = r * W + c
    
    # Compute capture mask for (2,0)
    groups_at = caps[:, r, c]
    valid = (groups_at >= 0)
    eq = (roots[:, :, None] == groups_at[:, None, :]) & valid[:, None, :]
    cap_mask_flat = eq.any(dim=2)
    cap_mask_flat[:, lin] = False
    cap_mask = cap_mask_flat.view(1, H, W)[0]
    
    print(f"Capture mask for playing at ({r},{c}):")
    for i in range(3):
        row = []
        for j in range(3):
            row.append('X' if cap_mask[i, j] else '.')
        print("  ", row)
    print()
    
    # Compute the post board manually
    post = tb.board[0].clone()
    post[r, c] = Stone.BLACK
    post = torch.where(cap_mask.to(torch.bool),
                       torch.tensor(Stone.EMPTY, dtype=torch.int8, device=tb.device),
                       post)
    
    print("Post board (after black plays at (2,0)):")
    for i in range(3):
        row = []
        for j in range(3):
            val = post[i, j].item()
            row.append(' B' if val == 0 else ' W' if val == 1 else ' .')
        print("  ", row)
    print()
    
    # Compute hashes
    repeat_hash = full_hash(tb, post)
    manual_hash = compute_hash_delta(tb, r, c, cap_mask)
    
    print(f"\nHash values:")
    print(f"  Current (pre) hash: {pre_hash}")
    print(f"  Expected post hash (full recompute): {repeat_hash}")
    print(f"  Manual delta hash: {manual_hash}")
    print(f"  Match? {repeat_hash == manual_hash}")
    
    # Now test the engine's computation
    print("\n--- Testing engine's filter_super_ko ---\n")
    
    # Seed history with the repeat hash
    tb.hash_history.zero_()
    tb.hash_history[0, 0] = repeat_hash
    tb.hash_history[0, 1] = pre_hash
    tb.move_count[0] = 2
    tb._invalidate_cache()
    
    # Get raw legal moves
    raw_legal, cap_info2 = tb.legal_checker.compute_legal_moves_with_captures(
        board=tb.board, current_player=tb.current_player, return_capture_info=True
    )
    
    # Apply super-ko filter
    filtered = tb._filter_super_ko_vectorized(raw_legal, cap_info2)
    
    print(f"Raw legal at ({r},{c}): {bool(raw_legal[0, r, c].item())}")
    print(f"Filtered legal at ({r},{c}): {bool(filtered[0, r, c].item())}")
    print(f"History: {[int(tb.hash_history[0, i].item()) for i in range(2)]}")
    
    # Debug the engine's hash computation
    B = 1
    N2 = H * W
    player = tb.current_player.long()
    Z = tb.Zpos
    cand_idx = torch.arange(N2, device=tb.device).view(1, -1)
    
    # Engine's placement delta
    place_old = Z[cand_idx, torch.zeros_like(player)[:, None]]
    place_new = Z[cand_idx, (player + 1)[:, None]]
    place_delta = torch.bitwise_xor(place_old, place_new)
    
    print(f"\nEngine's placement delta at position {lin}:")
    print(f"  place_delta[0, {lin}]: {int(place_delta[0, lin].item())}")
    print(f"  Manual place_delta: {int((Z[lin, 0] ^ Z[lin, player[0] + 1]).item())}")
    
    # Let's also check if the capture computation matches
    print("\nChecking capture computation...")
    
    # Simulate what the filter does
    roots2 = cap_info2["roots"]
    cg = cap_info2["capture_groups"]
    r_all = cand_idx // W
    c_all = cand_idx % W
    groups_at_all = cg[torch.arange(B, device=tb.device)[:, None], r_all, c_all]
    
    print(f"  Capture groups at ({r},{c}): {groups_at_all[0, lin].tolist()}")
    
    # Check if capture mask matches
    valid_any = (groups_at_all >= 0).any(dim=2, keepdim=True)
    groups_clamped = groups_at_all.clamp_min(0)
    eq_all = (roots2[:, None, :, None] == groups_clamped[:, :, None, :])
    cap_mask_all = eq_all.any(dim=3)
    cap_mask_all = cap_mask_all & valid_any.expand(-1, -1, cap_mask_all.size(2))
    
    cap_mask_at_lin = cap_mask_all[0, lin]
    print(f"  Engine's capture mask for candidate at ({r},{c}): {cap_mask_at_lin.nonzero(as_tuple=True)[0].tolist()}")
    print(f"  Manual capture mask: {cap_mask.view(-1).nonzero(as_tuple=True)[0].tolist()}")

if __name__ == "__main__":
    main()
