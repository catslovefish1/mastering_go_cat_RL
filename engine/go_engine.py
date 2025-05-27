# --------------------------------------------------------
from __future__ import annotations
import os, torch, torch.nn.functional as F          # ← reordered merely for PEP8

# (keep this env-var line if you still want the fallback on older macOS)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ─── device helper ────────────────────────────────────────────────────
def _select_device() -> torch.device:
    """Pick the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
# ---------------------------------------------------------------------

BLACK, WHITE, EMPTY = 0, 1, -1


class TensorBoard:
    """Batch-friendly Go engine implemented entirely with PyTorch tensors."""
    # ------------------------------------------------------------------ #
    def __init__(self,
                 batch_size: int = 1,
                 board_size: int = 19,
                 device: torch.device | str | None = None):        # ★ NEW arg
        # choose device (caller can override)
        self.device = torch.device(device) if device is not None else _select_device()

        self.batch_size  = batch_size
        self.board_size  = board_size

        # (B,2,H,W); uint8 is fine for 0/1 flags
        self.stones = torch.zeros((batch_size, 2, board_size, board_size),
                                  dtype=torch.uint8, device=self.device)

        self._init_zobrist()
        self._init_kernels()

        self.max_groups = board_size * board_size
        self.group_ids = torch.zeros((batch_size, board_size, board_size),
                                     dtype=torch.int32, device=self.device)
        self.group_liberties = torch.zeros((batch_size, self.max_groups),
                                           dtype=torch.int16, device=self.device)

        self.current_player = torch.zeros(batch_size, dtype=torch.uint8, device=self.device)
        self.current_hash   = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        self.ko_points      = torch.full((batch_size, 2), -1, dtype=torch.int16, device=self.device)
        self.pass_count     = torch.zeros(batch_size, dtype=torch.uint8, device=self.device)

    # ------------------------------------------------------------------ #
    def _init_zobrist(self):
        torch.manual_seed(42)
        max64 = torch.iinfo(torch.int64).max
        self.zobrist_keys = torch.randint(0, max64,
                                          (2, self.board_size, self.board_size),
                                          dtype=torch.int64, device=self.device)
        self.turn_keys    = torch.randint(0, max64, (2,),
                                          dtype=torch.int64, device=self.device)

    def _init_kernels(self):
        self.neighbor_kernel = torch.tensor([[0,1,0],[1,0,1],[0,1,0]],
                                            dtype=torch.float32, device=self.device)
        self.connect_kernel  = torch.tensor([[1,1,1],[1,0,1],[1,1,1]],
                                            dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    def get_empty_mask(self) -> torch.Tensor:
        return (self.stones[:, 0] == 0) & (self.stones[:, 1] == 0)

    def get_legal_moves_mask(self) -> torch.Tensor:
        empty = self.get_empty_mask()
        liberty = F.conv2d(empty.float().unsqueeze(1),
                           self.neighbor_kernel.unsqueeze(0).unsqueeze(0),
                           padding=1).squeeze(1)
        legal = empty & (liberty > 0)

        # -- handle ko points ------------------------------------------------
        ko = self.ko_points[:, 0] >= 0
        if ko.any():
            b = torch.arange(self.batch_size, device=self.device)[ko]
            r = self.ko_points[ko, 0]
            c = self.ko_points[ko, 1]
            # *** FIX: cast r and c to long before advanced indexing ***
            legal[b, r.long(), c.long()] = False
        # --------------------------------------------------------------------

        legal |= self._get_capture_moves_mask(empty)
        return legal

    def _get_capture_moves_mask(self, empty: torch.Tensor) -> torch.Tensor:
        B = self.batch_size
        opp = self.stones[torch.arange(B, device=self.device),
                          (1 - self.current_player).long()]

        liberty = F.conv2d(empty.float().unsqueeze(1),
                           self.neighbor_kernel.unsqueeze(0).unsqueeze(0),
                           padding=1).squeeze(1)
        weak = (opp > 0) & (liberty <= 1)

        adj = F.conv2d(weak.float().unsqueeze(1),
                       self.neighbor_kernel.unsqueeze(0).unsqueeze(0),
                       padding=1).squeeze(1) > 0
        return empty & adj

    # ------------------------------------------------------------------ #
    def place_stones_batch(self, pos: torch.Tensor):
        batch = torch.arange(self.batch_size, device=self.device)
        is_pass = pos[:, 0] < 0
        self.pass_count += is_pass.to(torch.uint8)
        self.pass_count[~is_pass] = 0
        self.ko_points[:] = -1

        play = ~is_pass
        if play.any():
            b  = batch[play]
            r  = pos[play, 0]; c = pos[play, 1]
            pl = self.current_player[play]

            self.stones[b, pl.long(), r, c] = 1
            self.current_hash[b] ^= self.zobrist_keys[pl.long(),
                                                      r.long(),
                                                      c.long()]
            self._process_captures_batch(b, r, c, pl)

        self.current_hash ^= self.turn_keys[self.current_player.long()]
        self.current_player = 1 - self.current_player
        self.current_hash ^= self.turn_keys[self.current_player.long()]

    def _process_captures_batch(self, b, r, c, pl):
        opp = (1 - pl).long()
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            rr, cc = r+dx, c+dy
            ok = (rr>=0)&(rr<self.board_size)&(cc>=0)&(cc<self.board_size)
            if not ok.any(): continue
            vb, vr, vc, vo = b[ok], rr[ok], cc[ok], opp[ok]
            if (self.stones[vb, vo, vr, vc] > 0).any():
                self._check_and_remove_captured_groups(vb, vr, vc, vo)

    def _check_and_remove_captured_groups(self, b, r, c, col):
        for i in range(b.numel()):
            B, R, C, CL = int(b[i]), int(r[i]), int(c[i]), int(col[i])
            if self.stones[B, CL, R, C] == 0: continue
            grp = self._flood_fill_group(B,R,C,CL)
            libs = (F.conv2d(grp.float().unsqueeze(0).unsqueeze(0),
                             self.neighbor_kernel.unsqueeze(0).unsqueeze(0),
                             padding=1).squeeze() *
                    self.get_empty_mask()[B].float()).sum()
            if libs == 0:
                self.stones[B, CL] &= (~grp).to(torch.uint8)
                capt = grp.nonzero(as_tuple=False)
                if capt.numel():
                    rows, cols = capt[:,0], capt[:,1]
                    self.current_hash[B] ^= torch.sum(self.zobrist_keys[CL, rows, cols])
                if capt.shape[0] == 1: self.ko_points[B] = capt[0]

    def _flood_fill_group(self,B,r,c,col):
        board = self.stones[B,col]
        seen  = torch.zeros_like(board, dtype=torch.bool, device=self.device)
        stack = [(r,c)]
        while stack:
            rr,cc = stack.pop()
            if rr<0 or rr>=self.board_size or cc<0 or cc>=self.board_size: continue
            if seen[rr,cc] or board[rr,cc]==0: continue
            seen[rr,cc]=True
            stack.extend([(rr+1,cc),(rr-1,cc),(rr,cc+1),(rr,cc-1)])
        return seen

    # ------------------------------------------------------------------ #
    def get_scores(self):


        b = self.stones[:,BLACK].sum((1,2)).float()
        w = self.stones[:,WHITE].sum((1,2)).float()
        return torch.stack([b,w],1)
    def is_game_over(self): return self.pass_count >= 2

    def to_features(self):
        ids = torch.arange(self.batch_size, device=self.device)
        cur = self.stones[ids, self.current_player.long()].float()
        opp = self.stones[ids, (1-self.current_player).long()].float()
        legal = self.get_legal_moves_mask().float()
        empty = self.get_empty_mask().float()
        cur_lib = (F.conv2d(cur.unsqueeze(1),
                            self.neighbor_kernel.unsqueeze(0).unsqueeze(0),
                            padding=1).squeeze(1)*empty)
        turn = self.current_player.view(-1,1,1).expand(-1,self.board_size,self.board_size).float()
        return torch.stack([cur,opp,legal,cur_lib,turn],1)


