import numpy as np
import yaml
import tensorflow as tf

from src.data_blobs import make_blobs

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg("configs/blobs.yaml")
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    X, y = make_blobs(n=cfg["n_samples"], seed=cfg["seed"])

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(cfg["n_samples"], seed=cfg["seed"]).batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(cfg["hidden"], activation="relu"),
        tf.keras.layers.Dense(cfg["hidden"], activation="relu"),
        tf.keras.layers.Dense(1)  # logits
    ])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb[:, None], logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(cfg["epochs"]):
        losses = []
        for xb, yb in ds:
            loss = train_step(xb, yb)
            losses.append(loss.numpy())
        # quick accuracy on full data
        logits = model(X, training=False).numpy().reshape(-1)
        pred = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.float32)
        acc = (pred == y).mean()
        print(f"TF Epoch {epoch+1:02d} | loss={float(np.mean(losses)):.4f} | acc={acc:.4f}")

if __name__ == "__main__":
    main()
