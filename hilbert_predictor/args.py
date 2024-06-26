import argparse

import torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simple",
    action="store_true",
    help='Load data only from the "arc-datasets/datasets/bitdata/training" directory',
)
parser.add_argument(
    "--wandb", action="store_true", help="Enable Weights & Biases logging"
)

batch_size = 4
if torch.cuda.is_available():
    batch_size = 64
parser.add_argument(
    "--batch_size", type=int, default=batch_size, help="Batch size for training"
)
parser.add_argument(
    "--dropout_rate", type=float, default=0.1, help="Dropout rate for the model"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=Path("checkpoint.pt"),
    help="Path to save the model checkpoint",
)
parser.add_argument(
    "--hilbert", type=bool, default=False, help="Use Hilbert curve for data processing"
)
parser.add_argument(
    "--quadtree", type=bool, default=False, help="Use quadtree position encoding"
)
parser.add_argument(
    "--schedulefree", type=bool, default=False, help="Use schedule free optimization"
)
print("Parser args: ", parser.parse_args())

args = parser.parse_args()

batch_size = args.batch_size
dropout_rate = args.dropout_rate
checkpoint_path = args.checkpoint_path
hilbert = args.hilbert
quadtree = args.quadtree
use_schedulefree = args.schedulefree
