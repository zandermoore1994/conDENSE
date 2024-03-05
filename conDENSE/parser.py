"""Arg parser for runner script."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,  default='data')
parser.add_argument("--n_epochs", type=str,  default=50)
parser.add_argument("--batch_size", type=int,  default=128)
parser.add_argument("--window_size", type=int,  default=20)
parser.add_argument("--univariate", action=argparse.BooleanOptionalAction)
args = parser.parse_args()