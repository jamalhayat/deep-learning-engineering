import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
import torch
from src.data_blobs import make_blobs


def main():
    X, y = make_blobs(n=32, seed=0)

    # TensorFlow
    tf_out = tf.keras.layers.Dense(3)(X)

    # PyTorch
    pt_out = torch.nn.Linear(2, 3)(torch.tensor(X))

    print("TF OK:", tf_out.shape)
    print("PT OK:", pt_out.shape)
    print("Data y unique:", sorted(set(y.tolist())))


if __name__ == "__main__":
    main()

