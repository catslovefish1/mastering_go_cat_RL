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

# ===========================================================================
#  TensorBoard  –  *vectorised* capture logic (no Python loops / .item() sync)
# ===========================================================================
class TensorBoard:
    """Batch‑friendly Go engine implemented entirely with PyTorch tensors.

    Main change vs. the earlier version: all capture detection & ko handling
    is now done in pure tensor code (see `_capture_mask`).  The three former
    hotspots `_process_captures_batch`, `_check_and_remove_captured_groups`,
    `_flood_fill_group` are gone.
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 batch_size: int = 1,
                 board_size: int = 19,
                 device: torch.device | str | None = None):
        self.device = torch.device(device) if device is not None else _select_device()
        self.batch_size, self.board_size = batch_size, board_size

        # Stones tensor (B, 2, H, W) – uint8 is enough for 0/1 flags
        self.stones = torch.zeros((batch_size, 2, board_size, board_size),
                                  dtype=torch.uint8, device=self.device)

        self._init_zobrist()
        self._init_kernels()

        self.current_player = torch.zeros(batch_size, dtype=torch.uint8, device=self.device)
        self.current_hash   = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        self.ko_points      = torch.full((batch_size, 2), -1, dtype=torch.int16, device=self.device)
        self.pass_count     = torch.zeros(batch_size, dtype=torch.uint8, device=self.device)

    # ------------------------------------------------------------------ #
    def _init_zobrist(self):
        torch.manual_seed(42)                       # reproducible keys
        max64 = torch.iinfo(torch.int64).max
        self.zobrist_keys = torch.randint(0, max64,
                                          (2, self.board_size, self.board_size),
                                          dtype=torch.int64, device=self.device)
        self.turn_keys    = torch.randint(0, max64, (2,),
                                          dtype=torch.int64, device=self.device)

    def _init_kernels(self):
        self.neighbor_kernel = torch.tensor([[0,1,0], [1,0,1], [0,1,0]],
                                            dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    #   High‑level helpers
    # ------------------------------------------------------------------ #
    def get_empty_mask(self) -> torch.Tensor:
        """Return (B,H,W) bool mask where board is empty."""
        return (self.stones[:, 0] == 0) & (self.stones[:, 1] == 0)

    def get_scores(self):
        b = self.stones[:, BLACK].sum((1, 2)).float()
        w = self.stones[:, WHITE].sum((1, 2)).float()
        return torch.stack([b, w], 1)

    def is_game_over(self):
        return self.pass_count >= 2

    # ------------------------------------------------------------------ #
    #   Legal‑move mask (unchanged – still uses simple conv trick)
    # ------------------------------------------------------------------ #
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

    def get_legal_moves_mask(self) -> torch.Tensor:
        empty = self.get_empty_mask()
        liberty = F.conv2d(empty.float().unsqueeze(1),
                           self.neighbor_kernel.unsqueeze(0).unsqueeze(0),
                           padding=1).squeeze(1)
        legal = empty & (liberty > 0)

        # ko point mask --------------------------------------------------
        ko = self.ko_points[:, 0] >= 0
        if ko.any():
            b = torch.arange(self.batch_size, device=self.device)[ko]
            r = self.ko_points[ko, 0].long()
            c = self.ko_points[ko, 1].long()
            legal[b, r, c] = False

        legal |= self._get_capture_moves_mask(empty)
        return legal

    # ------------------------------------------------------------------ #
    #   *** Vectorised capture detection + ko handling ***
    # ------------------------------------------------------------------ #
    def _capture_mask(self, played_mask: torch.Tensor, pl: torch.Tensor):
        """Return (captured_mask, ko_coords) after `played_mask` (B,H,W) applied.

        * `played_mask` – bool tensor with exactly the just‑placed stones = 1.
        * `pl`          – (B,) current player colours (0/1) for each board.
        """
        B, H, W = played_mask.shape
        opp_pl  = 1 - pl[:, None, None]                  # broadcast → (B,1,1)

        opp_stones = self.stones[torch.arange(B, device=self.device), opp_pl] > 0

        empty = self.get_empty_mask()
        nk    = self.neighbor_kernel[None, None]         # (1,1,3,3)

        # Opponent liberty count **after** the move
        opp_lib = F.conv2d(opp_stones.float().unsqueeze(1), nk, padding=1)
        opp_lib = (opp_lib.squeeze(1) * empty.float())

        # Opponent stones now with <=0 libs are candidates for capture
        mask = opp_stones & (opp_lib == 0)

        # Dilate mask over connected opponent stones until fixed point
        while True:
            grown = (F.conv2d(mask.float().unsqueeze(1), nk, padding=1).squeeze(1) > 0) & opp_stones
            new_mask = mask | grown
            if (new_mask ^ mask).sum() == 0:
                break
            mask = new_mask

        # Re‑check liberties to exclude strings that actually still have >0
        lib2 = F.conv2d(mask.float().unsqueeze(1), nk, padding=1).squeeze(1)
        lib2 = lib2 * empty.float()
        captured = mask & (lib2 == 0)

        # ko: single‑stone capture → remember its coords
        ko_rowcol = torch.full((B, 2), -1, dtype=torch.int16, device=self.device)
        single = captured.sum((1, 2)) == 1
        if single.any():
            coords = captured[single].nonzero(as_tuple=False)  # (N,3): batch,row,col
            # keep last two columns; they already correspond to rows/cols
            ko_rowcol[single] = coords[:, 1:]
        return captured, ko_rowcol

    # ------------------------------------------------------------------ #
    #   Main game‑state mutation   (vectorised)
    # ------------------------------------------------------------------ #
    def place_stones_batch(self, pos: torch.Tensor):
        """`pos` is (B,2) int32 with row,col or <-1,-1> for pass."""
        batch_idx = torch.arange(self.batch_size, device=self.device)
        is_pass   = pos[:, 0] < 0

        # update consecutive‑pass counter & ko table
        self.pass_count += is_pass.to(torch.uint8)
        self.pass_count[~is_pass] = 0
        self.ko_points[:] = -1

        play_mask = ~is_pass
        if play_mask.any():
            b  = batch_idx[play_mask]
            r  = pos[play_mask, 0]
            c  = pos[play_mask, 1]
            pl = self.current_player[play_mask]

            # --- place the stones -------------------------------------
            self.stones[b, pl.long(), r, c] = 1
            self.current_hash[b] ^= self.zobrist_keys[pl.long(), r.long(), c.long()]

            # bool mask of just‑played stones on the full (B,H,W) grid
            played = torch.zeros_like(self.stones[:, 0], dtype=torch.bool)
            played[b, r, c] = True

            # --- capture detection & ko -------------------------------
            capt, ko = self._capture_mask(played, self.current_player)
            if capt.any():
                # remove captured stones from board tensor
                self.stones &= (~capt).unsqueeze(1).to(torch.uint8)

                # Zobrist update per stone (small Python loop OK)
                nz = capt.nonzero(as_tuple=False)           # (N,3)
                if nz.numel():
                    rows, cols = nz[:, 1], nz[:, 2]
                    colours = (1 - self.current_player[nz[:, 0]]).long()
                    keys = self.zobrist_keys[colours, rows, cols]
                    # XOR per batch element
                    for idx in range(nz.shape[0]):
                        self.current_hash[nz[idx, 0]] ^= keys[idx]

            self.ko_points = ko

        # --- hand over the turn --------------------------------------
        self.current_hash ^= self.turn_keys[self.current_player.long()]
        self.current_player = 1 - self.current_player
        self.current_hash ^= self.turn_keys[self.current_player.long()]

    # ------------------------------------------------------------------ #
    #   Extra: 5‑plane AlphaZero‑style feature extractor (unchanged)
    # ------------------------------------------------------------------ #
    def to_features(self):
        ids    = torch.arange(self.batch_size, device=self.device)
        cur    = self.stones[ids, self.current_player.long()].float()
        opp    = self.stones[ids, (1 - self.current_player).long()].float()
        legal  = self.get_legal_moves_mask().float()
        empty  = self.get_empty_mask().float()
        lib    = F.conv2d(cur.unsqueeze(1),
                          self.neighbor_kernel.unsqueeze(0).unsqueeze(0),
                          padding=1).squeeze(1) * empty
        turn_p = self.current_player.view(-1, 1, 1).expand(-1, self.board_size, self.board_size).float()
        return torch.stack([cur, opp, legal, lib, turn_p], 1)

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

    # ------------------------------------------------------------------ #
    def _flood_fill_group(self, B: int, r: int, c: int, col: int) -> torch.Tensor:
        """
        GPU flood-fill using iterative 3×3 dilation.
        Returns a (H×W) bool mask for the group containing (r, c).
        """
        board = self.stones[B, col].bool()                # (H, W)
        if not board[r, c]:
            return torch.zeros_like(board)

        kernel = self.neighbor_kernel.unsqueeze(0).unsqueeze(0)    # (1,1,3,3)

        grp = torch.zeros_like(board, dtype=torch.bool, device=self.device)
        grp[r, c] = True
        grp    = grp.unsqueeze(0).unsqueeze(0).float()     # (1,1,H,W)
        board4 = board.unsqueeze(0).unsqueeze(0)           # (1,1,H,W)

        while True:
            nbr     = (F.conv2d(grp, kernel, padding=1) > 0) & board4
            new_grp = grp.bool() | nbr
            if new_grp.equal(grp.bool()):                  # converged
                return new_grp.squeeze(0).squeeze(0)       # (H, W)
            grp = new_grp.float()

    

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



# ===================================================================== #
class TensorBatchBot:
    def __init__(self, device: torch.device | str | None = None):   # ★ NEW arg
        self.device = torch.device(device) if device is not None else _select_device()
    def select_moves(self, boards: TensorBoard):
        legal = boards.get_legal_moves_mask()
        B,H,W = legal.shape
        flat = legal.view(B,-1)
        moves = torch.full((B,2), -1, dtype=torch.int32, device=self.device)
        play  = flat.any(1)
        if play.any():
            probs = flat[play].float(); probs/=probs.sum(1,keepdim=True)
            idx = torch.multinomial(probs,1).squeeze(1).to(torch.int32)
            moves[play,0] = idx // W
            moves[play,1] = idx %  W
        return moves

