"""
PyTorch training script for a simple MLP on a synthetic 2D
binary classification task.

Key characteristics:
- Config-driven via YAML
- Explicit training loop using autograd
- Logit-based loss for numerical stability
- Minimal structure intended for extensibility
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_blobs import make_blobs


"""Load experiment configuration."""
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


"""Select the best available compute device (CUDA, MPS, or CPU)."""
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


"""Simple MLP model that outputs logits."""
class MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # logits
        )

    def forward(self, x):
        return self.net(x)


# Training pipeline
def main():
    # Load experiment parameters
    cfg = load_cfg("configs/blobs.yaml")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Select runtime device
    device = get_device()
    print("Device:", device)

    # Generate synthetic data
    X, y = make_blobs(n=cfg["n_samples"], seed=cfg["seed"])
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    # Create dataset pipeline
    dl = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    # Define model architecture
    model = MLP(hidden=cfg["hidden"]).to(device)

    # Define loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Epoch loop
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_losses = []

        # Mini-batch training
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())

        # Lightweight evaluation on training data
        model.eval()
        with torch.no_grad():
            logits = model(X_t.to(device)).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds.cpu() == y_t).float().mean().item()

        # Metrics reporting
        print(
            f"PT Epoch {epoch + 1:02d} | "
            f"loss={float(np.mean(epoch_losses)):.4f} | "
            f"acc={acc:.4f}"
        )


if __name__ == "__main__":
    main()
