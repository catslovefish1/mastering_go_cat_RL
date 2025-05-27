from __future__ import annotations
"""Light‑weight random (uniform) move generator for batched TensorBoard engines.

Placed in the *agents* package so other scripts (e.g. duel_random_vs_random.py)
can simply do

    from agents.basic import TensorBatchBot

The implementation stays device‑agnostic and works with CPU, CUDA or Apple MPS.
"""

import torch
from torch import Tensor
from engine.go_engine import TensorBoard

# ---------------------------------------------------------------------------
# Device helper – keep a local copy so the agent is self‑contained
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    """Pick best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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

