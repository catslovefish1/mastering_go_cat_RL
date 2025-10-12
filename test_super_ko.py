#!/usr/bin/env python3
from __future__ import annotations
import torch
# Adjust these imports to match your actual filenames:
from engine.GoLegalMoveChecker import GoLegalMoveChecker
from engine.tensor_native_stable import TensorBoard, Stone   # change module name if needed

def xor_reduce_last_dim(x: torch.Tensor) -> torch.Tensor:
    while x.size(-1) > 1:
        n = x.size(-1)
        x = torch.bitwise_xor(x[..., : n//2], x[..., n//2 : (n//2)*2])
        if n % 2:
            x = torch.bitwise_xor(x, x[..., -1:].expand_as(x))
    return x.squeeze(-1)

def full_hash(tb: TensorBoard, board_2d: torch.Tensor) -> int:
    H=W=tb.board_size
    idx = (board_2d.to(torch.long)+1).view(-1)  # -1,0,1 -> 0,1,2
    lin = torch.arange(H*W, device=tb.device, dtype=torch.long)
    z = tb.Zpos[lin, idx]
    # 1D reduce
    x = z
    while x.numel() > 1:
        n = x.numel()
        x = torch.bitwise_xor(x[: n//2], x[n//2 : (n//2)*2])
        if n % 2:
            x = torch.bitwise_xor(x, x[-1:].expand_as(x))
    return int(x.item())

def build_tb(pre_flat, color="black"):
    tb = TensorBoard(batch_size=1, board_size=3, enable_super_ko=True, enable_timing=False)
    H=W=3
    b = torch.tensor(pre_flat, dtype=torch.int8, device=tb.device).view(H,W)
    tb.board[0]=b
    tb.current_player[0]=0 if color=="black" else 1
    tb.current_hash[0]=full_hash(tb,b)
    tb._invalidate_cache()
    return tb

def main():
    # Pre board = move 31 POST (so (2,0) is EMPTY)
    pre_flat = [
        0,  0, -1,
        1,  0,  1,
       -1,  1,  1,
    ]
    r,c = 2,0

    tb = build_tb(pre_flat, "black")

    # --- derive the would-be post board for recapture to seed history ---
    legal0 = tb.legal_moves()  # fills _last_capture_info
    cap = tb._last_capture_info
    roots = cap["roots"]                       # (B,N2)
    caps = cap["capture_groups"]               # (B,H,W,4)

    H=W=tb.board_size
    lin = r*W+c
    groups_at = caps[:, r, c]                  # (B,4)
    valid = (groups_at >= 0)
    eq = (roots[:, :, None] == groups_at[:, None, :]) & valid[:, None, :]
    cap_mask_flat = eq.any(dim=2)              # (B,N2)
    cap_mask_flat[:, lin] = False
    cap_mask = cap_mask_flat.view(1,H,W)[0]

    post = tb.board[0].clone()
    post[r,c]=Stone.BLACK
    post = torch.where(cap_mask.to(torch.bool),
                       torch.tensor(Stone.EMPTY, dtype=torch.int8, device=tb.device),
                       post)
    repeat_hash = full_hash(tb, post)
    pre_hash = int(tb.current_hash[0].item())

    # --- seed history EXACTLY as engine expects ---
    tb.hash_history.zero_()
    tb.hash_history[0,0]=repeat_hash
    tb.hash_history[0,1]=pre_hash
    tb.move_count[0]=2
    tb._invalidate_cache()

    # --- recompute legal mask in three flavors ---
    # 1) raw legality (no super-ko) by calling the checker directly
    raw_legal, cap_info2 = tb.legal_checker.compute_legal_moves_with_captures(
        board=tb.board, current_player=tb.current_player, return_capture_info=True
    )
    # 2) filtered by calling private filter with the cap info from (1)
    filtered = tb._filter_super_ko_vectorized(raw_legal, cap_info2)

    # 3) reproduce filter math here to isolate
    B=1; N2=H*W
    cand_mask = raw_legal.view(B,N2).bool()
    player = tb.current_player.long()
    opp    = 1-player
    Z = tb.Zpos
    cand_idx = torch.arange(N2, device=tb.device).view(1,-1)
    place_old = Z[cand_idx, torch.zeros_like(player)[:,None]]
    place_new = Z[cand_idx, (player+1)[:,None]]
    place_delta = torch.bitwise_xor(place_old, place_new)

    roots2 = cap_info2["roots"]
    cg = cap_info2["capture_groups"]
    r_all = cand_idx // W
    c_all = cand_idx %  W
    groups_at_all = cg[torch.arange(B,device=tb.device)[:,None], r_all, c_all]  # (B,N2,4)
    valid_any = (groups_at_all >= 0).any(dim=2, keepdim=True)
    groups_clamped = groups_at_all.clamp_min(0)

    eq_all = (roots2[:,None,:,None] == groups_clamped[:,:,None,:]) # (B,N2,N2,4)
    cap_mask_all = eq_all.any(dim=3)
    cap_mask_all = cap_mask_all & valid_any.expand(-1,-1,cap_mask_all.size(2))
    cap_mask_all = cap_mask_all & cand_mask[:,:,None]

    ZT = Z.transpose(0,1)
    Z_emp = ZT[0].expand(B,-1)
    Z_opp = ZT[(opp+1)]

    sel_opp = torch.where(cap_mask_all, Z_opp[:,None,:], torch.zeros(1,1,N2, dtype=torch.long, device=tb.device))
    sel_emp = torch.where(cap_mask_all, Z_emp[:,None,:], torch.zeros(1,1,N2, dtype=torch.long, device=tb.device))
    cap_xor_opp = xor_reduce_last_dim(sel_opp)
    cap_xor_emp = xor_reduce_last_dim(sel_emp)
    cap_delta = torch.bitwise_xor(cap_xor_opp, cap_xor_emp)

    new_hash = (tb.current_hash[:,None] ^ place_delta) ^ cap_delta  # (B,N2)

    max_moves = tb.hash_history.shape[1]
    HIST = tb.hash_history
    hist_mask = torch.arange(max_moves, device=tb.device)[None,:] < tb.move_count[:,None]
    matches = (new_hash[:,:,None] == HIST[:,None,:]) & hist_mask[:,None,:]
    is_repeat_flat = matches.any(dim=2) & cand_mask
    repeat_mask = is_repeat_flat.view(B,H,W)

    # --- print a compact report ---
    print("\n=== SUPER-KO INTERNALS ===")
    print(f"pre_hash: {pre_hash}")
    print(f"repeat_hash: {repeat_hash}")
    print("hash_history[0:move_count]:",
          [int(tb.hash_history[0,i].item()) for i in range(int(tb.move_count[0].item()))])
    print(f"move_count: {int(tb.move_count[0].item())}")
    print(f"hist_mask True idxs: {[i for i in range(max_moves) if hist_mask[0,i].item()]}")
    print(f"raw legal at (2,0): {bool(raw_legal[0,2,0].item())}")
    print(f"filtered  at (2,0): {bool(filtered[0,2,0].item())}")
    print(f"repeat_msk at (2,0): {bool(repeat_mask[0,2,0].item())}")
    print(f"cand new_hash at (2,0): {int(new_hash.view(B,N2)[0, lin].item())}")
    print(f"matches at (2,0) vs history:",
          [(i, bool(matches.view(B,N2,-1)[0, lin, i].item())) for i in range(int(tb.move_count[0].item()))])

if __name__ == "__main__":
    main()
