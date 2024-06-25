import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--simple', action='store_true', help='Load data only from the "arc-datasets/datasets/bitdata/training" directory')
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
args = parser.parse_args()
