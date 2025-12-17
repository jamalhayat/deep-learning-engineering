"""
TensorFlow training script for a simple MLP on a synthetic 2D
binary classification task.

Key characteristics:
- Config-driven via YAML
- Explicit training loop using GradientTape
- Logit-based loss for numerical stability
- Minimal structure intended for extensibility
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import tensorflow as tf

from src.data_blobs import make_blobs

"""Load experiment configuration."""
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Training pipeline
def main():
    # Load experiment parameters
    cfg = load_cfg("configs/blobs.yaml")
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Generate synthetic data
    X, y = make_blobs(n=cfg["n_samples"], seed=cfg["seed"])

    # Create dataset pipeline
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(cfg["n_samples"], seed=cfg["seed"]).batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)

    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(cfg["hidden"], activation="relu"),
        tf.keras.layers.Dense(cfg["hidden"], activation="relu"),
        tf.keras.layers.Dense(1)  # logits
    ])

    # Define loss function and optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])

    # Define training step
    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb[:, None], logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Epoch loop
    for epoch in range(cfg["epochs"]):
        epoch_losses = []

        # Mini-batch training
        for xb, yb in ds:
            loss = train_step(xb, yb)
            epoch_losses.append(loss.numpy())

        # Lightweight evaluation on training data
        logits = model(X, training=False).numpy().reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(np.float32)
        acc = (preds == y).mean()

        # Metrics reporting
        print(
            f"TF Epoch {epoch + 1:02d} | "
            f"loss={float(np.mean(epoch_losses)):.4f} | "
            f"acc={acc:.4f}"
        )

if __name__ == "__main__":
    main()
