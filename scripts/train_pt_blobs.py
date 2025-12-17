import numpy as np
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_blobs import make_blobs

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)

def main():
    cfg = load_cfg("configs/blobs.yaml")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = get_device()
    print("Device:", device)

    X, y = make_blobs(n=cfg["n_samples"], seed=cfg["seed"])
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=cfg["batch_size"], shuffle=True)

    model = MLP(hidden=cfg["hidden"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        model.train()
        losses = []
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        # accuracy on full data
        model.eval()
        with torch.no_grad():
            logits = model(X_t.to(device)).squeeze(1)
            pred = (torch.sigmoid(logits) > 0.5).float()
            acc = (pred.cpu() == y_t).float().mean().item()

        print(f"PT Epoch {epoch+1:02d} | loss={float(np.mean(losses)):.4f} | acc={acc:.4f}")

if __name__ == "__main__":
    main()
