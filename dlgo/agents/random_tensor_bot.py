# uniform-random bot (passes only if no stone is playable)
import os, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class TensorRandomBot:
    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size

    def select_move(self, game_state):
        states = (game_state,) if self.batch_size is None else game_state
        legal_lists = []
        for st in states:
            mv = st.legal_moves()
            stones = [m for m in mv if m.is_play]
            legal_lists.append(stones or mv)          # keep pass only if forced

        max_len = max(len(lst) for lst in legal_lists)
        pad = -1
        idx = torch.full((len(states), max_len), pad, dtype=torch.long, device=DEVICE)
        for r,lst in enumerate(legal_lists):
            idx[r, :len(lst)] = torch.arange(len(lst), device=DEVICE)
        probs = (idx != pad).float()
        choice = torch.multinomial(probs, 1).squeeze(1)
        chosen = [legal_lists[r][int(c)] for r,c in enumerate(choice)]
        return chosen[0] if self.batch_size is None else chosen
