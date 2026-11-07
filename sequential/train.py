"""Train a sequential recommender (SASRec, GRU4Rec) on a frozen split.

    conda run -n pmrs python -m sequential.train --model SASRec --dataset ml-1m --seeds 0 1 2
"""
from common.recbole_runner import cli

if __name__ == "__main__":
    cli("sequential")
