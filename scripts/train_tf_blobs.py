"""
TensorFlow training script for a simple MLP on a synthetic 2D
binary classification task.

Key characteristics:
- Config-driven via YAML
- Explicit training loop using GradientTape
- Logit-based loss for numerical stability
- Minimal structure intended for extensibility
"""
import os
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    cfg = load_cfg("configs/blobs.yaml")
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Generate synthetic dataset
    X, y = make_blobs(n=cfg["n_samples"], seed=cfg["seed"])

    # Deterministic train/val split (framework-agnostic)
    n_val = int(cfg["n_samples"] * cfg["val_ratio"])
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    # Create dataset pipeline (train only; validation evaluated explicitly)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=cfg["seed"])
        .batch(cfg["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )

    # Define model architecture (outputs logits)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(cfg["hidden"], activation="relu"),
        tf.keras.layers.Dense(cfg["hidden"], activation="relu"),
        tf.keras.layers.Dense(1),
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
        train_losses = []

        for xb, yb in train_ds:
            loss = train_step(xb, yb)
            train_losses.append(loss.numpy())

        # Evaluate on validation split
        val_logits = model(X_val, training=False).numpy().reshape(-1)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_preds = (val_probs > 0.5).astype(np.float32)
        val_acc = (val_preds == y_val).mean()

        print(
            f"TF Epoch {epoch + 1:02d} | "
            f"train_loss={float(np.mean(train_losses)):.4f} | "
            f"val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
