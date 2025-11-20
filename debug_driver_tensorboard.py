# debug_driver_tensorboard.py
import torch
from engine.tensor_native import TensorBoard  # adjust import if needed

def mps_driver_mb():
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / (1024 ** 2)
    return 0.0

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    B = 1024
    board_size = 19

    tb = TensorBoard(
        batch_size=B,
        board_size=board_size,
        history_factor=1,
        device=device,
        enable_timing=False,
        enable_super_ko=True,   # toggle this to False to see super-ko effect
    )

    # Random positions for testing
    H = W = board_size
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    print(f"Using device: {device}")
    print(f"Initial MPS driver = {mps_driver_mb():.1f} MB")

    for ply in range(1, 201):
        # random moves, allow passes too
        rows = torch.randint(-1, H, (B,), generator=rng)
        cols = torch.randint(-1, W, (B,), generator=rng)
        positions = torch.stack([rows, cols], dim=1)

        # 1) just call legal_moves() and look at driver
        tb.legal_moves()
        drv_after_legal = mps_driver_mb()

        # 2) then step() and look again
        tb.step(positions)
        drv_after_step = mps_driver_mb()

        print(
            f"Ply {ply:4d}: "
            f"driver_after_legal={drv_after_legal:7.1f} MB, "
            f"driver_after_step={drv_after_step:7.1f} MB"
        )

if __name__ == "__main__":
    main()
