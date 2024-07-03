import argparse

import torch
from pathlib import Path
import sys

def is_running_under_pytest():
    return any("pytest" in arg for arg in sys.argv)

if not is_running_under_pytest():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kindergarten",
        action="store_true",
        help='Load data only from the "arc-datasets/datasets/kindergarten/data/training" directory',
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
        "--hilbert", action="store_true", help="Use Hilbert curve for data processing"
    )
    parser.add_argument(
        "--quadtree", action="store_true", help="Use use_quadtree position encoding"
    )
    parser.add_argument(
        "--schedulefree", action="store_true", help="Use schedule free optimization"
    )
    parser.add_argument(
        "--grokfast", action="store_true", help="Use Grokfast optimization"
    )
    print("Parser args: ", parser.parse_args())

    args = parser.parse_args()
    
    args = {
        "kindergarten": args.kindergarten,
        "wandb": args.wandb,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "checkpoint_path": args.checkpoint_path,
        "hilbert": args.hilbert,
        "quadtree": args.quadtree,
        "schedulefree": args.schedulefree,
        "grokfast": args.grokfast,
    }
else:
    args = {
        "kindergarten": True,
        "wandb": False,
        "batch_size": 4,
        "dropout_rate": 0.1,
        "checkpoint_path": "checkpoint.pt",
        "hilbert": True,
        "quadtree": True,
        "schedulefree": False,
        "grokfast": True,
    }

use_wandb = args.get('wandb', False)
kindergarten = args.get('kindergarten', True)
batch_size = args.get('batch_size', 4)
dropout_rate = args.get('dropout_rate', 0.1)
checkpoint_path = args.get("checkpoint_path", "checkpoint.pt")
use_hilbert = args.get('hilbert', False)
use_quadtree = args.get('quadtree', False)
use_schedulefree = args.get('schedulefree', False)
use_grokfast = args.get('grokfast', False)
