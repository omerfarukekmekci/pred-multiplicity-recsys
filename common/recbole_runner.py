"""Train a RecBole model on a frozen split and export its rank lists.

Each family folder (sequential/, graph/) holds configs/family.yaml, one
configs/<Model>.yaml per model, and a thin train.py that calls cli().
Config files are merged in this order, later ones winning:

    common/configs/base.yaml -> <family>/configs/family.yaml -> <family>/configs/<Model>.yaml

RecBole only trains the model and picks the best epoch on validation. The
exported results are computed here, in the same way for every family: score
all items, remove the user's history items, then rank. (RecBole itself masks
history for general models but not for sequential ones, which would make rank
lists from the two families incomparable.)

Each run writes to <family>/runs/<dataset>/<Model>/seed<seed>/:

    run_info.json            model, dataset, seed, versions, git commit, resolved config
    metrics.json             mean metrics per split
    {valid,test}_users.csv   per-user target item, rank and metrics
    {valid,test}_topk.npy    top-K item ids per user (rows follow *_users.csv)
    *.pth                    best checkpoint
"""
import os

# Must be set before CUDA is initialised for cuBLAS to run deterministically.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
import recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed

from common.paths import ROOT
from common.recbole_data import ensure_recbole_files, load_split

FORMATS = {"sequential": "sequential", "graph": "general"}


def config_files(family, model):
    return [ROOT / "common" / "configs" / "base.yaml",
            ROOT / family / "configs" / "family.yaml",
            ROOT / family / "configs" / f"{model}.yaml"]


def history_matrix(dataset, frames):
    """Sparse (user x item) matrix of the given interactions, in RecBole's internal ids."""
    df = pd.concat(frames)
    users = dataset.token2id(dataset.uid_field, df["user"].astype(str).tolist())
    items = dataset.token2id(dataset.iid_field, df["item"].astype(str).tolist())
    return sp.csr_matrix((np.ones(len(df), dtype=bool), (users, items)),
                         shape=(dataset.user_num, dataset.item_num))


@torch.no_grad()
def export_split(model, dataset, eval_data, history, ks, save_k, device):
    model.eval()
    users, targets, ranks, topk = [], [], [], []
    for interaction, _, positive_u, positive_i in eval_data:
        interaction = interaction.to(device)
        scores = model.full_sort_predict(interaction).view(-1, dataset.item_num)
        uids = interaction[dataset.uid_field].cpu().numpy()
        # Leave-one-out gives exactly one target per user.
        assert len(positive_u) == len(uids)
        target = torch.empty(len(uids), dtype=torch.long, device=device)
        target[positive_u.to(device)] = positive_i.to(device)

        scores[:, 0] = -torch.inf  # RecBole's padding item
        hist = history[uids].tocoo()
        scores[torch.as_tensor(hist.row, device=device), torch.as_tensor(hist.col, device=device)] = -torch.inf

        target_score = scores.gather(1, target[:, None])
        assert torch.isfinite(target_score).all(), "a target item was masked as history"
        ranks.append(((scores > target_score).sum(1) + 1).cpu())
        topk.append(torch.topk(scores, save_k, dim=1).indices.cpu())
        users.append(uids)
        targets.append(target.cpu())

    to_raw = lambda field, ids: dataset.id2token(field, ids).astype(np.int64)
    rank = torch.cat(ranks).numpy()
    per_user = pd.DataFrame({
        "user": to_raw(dataset.uid_field, np.concatenate(users)),
        "target": to_raw(dataset.iid_field, torch.cat(targets).numpy()),
        "rank": rank,
        "mrr": 1.0 / rank,
    })
    for k in ks:
        per_user[f"hit@{k}"] = (rank <= k).astype(float)
        per_user[f"ndcg@{k}"] = np.where(rank <= k, 1.0 / np.log2(rank + 1), 0.0)
    topk = to_raw(dataset.iid_field, torch.cat(topk).numpy())
    order = np.argsort(per_user["user"].values, kind="stable")
    return per_user.iloc[order].reset_index(drop=True), topk[order]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(family, model_name, dataset_name, seed, overrides=None):
    files = config_files(family, model_name)
    settings = {}
    for f in files:
        settings.update(yaml.safe_load(f.read_text()) or {})
    settings.update(overrides or {})
    data_path, rb_dataset = ensure_recbole_files(
        dataset_name, FORMATS[family], settings.get("MAX_ITEM_LIST_LENGTH", 50))

    run_dir = ROOT / family / "runs" / dataset_name / model_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = Config(model=model_name, dataset=rb_dataset, config_file_list=[str(f) for f in files],
                    config_dict={**(overrides or {}), "data_path": str(data_path), "seed": seed,
                                 "checkpoint_dir": str(run_dir)})
    init_seed(config["seed"], config["reproducibility"])
    torch.use_deterministic_algorithms(config["deterministic_algorithms"], warn_only=True)
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    # Importing RecBole's SINE model switches on autograd anomaly detection for the
    # whole process, which makes training ~10x slower for every model.
    torch.autograd.set_detect_anomaly(False)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    start = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"])
    train_seconds = time.time() - start

    # RecBole's own reload calls torch.load without weights_only=False, which
    # torch >= 2.6 rejects for checkpoints that contain the config object.
    checkpoint = torch.load(trainer.saved_model_file, map_location=config["device"], weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    split = load_split(dataset_name)
    histories = {"valid": [split["train"]], "test": [split["train"], split["valid"]]}
    metrics = {}
    for name, loader in (("valid", valid_data), ("test", test_data)):
        per_user, topk = export_split(model, dataset, loader, history_matrix(dataset, histories[name]),
                                      config["export_ks"], config["export_topk"], config["device"])
        per_user.to_csv(run_dir / f"{name}_users.csv", index=False)
        np.save(run_dir / f"{name}_topk.npy", topk.astype(np.int32))
        metrics[name] = per_user.drop(columns=["user", "target", "rank"]).mean().to_dict()

    metrics["recbole_best_valid"] = {"score": best_valid_score, **dict(best_valid_result)}
    metrics["train_seconds"] = train_seconds
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "run_info.json").write_text(json.dumps({
        "family": family, "model": model_name, "dataset": dataset_name, "seed": seed,
        "best_epoch_checkpoint": os.path.basename(trainer.saved_model_file),
        "git_commit": git_commit(),
        "versions": {"recbole": recbole.__version__, "torch": torch.__version__, "numpy": np.__version__},
        "config": config.final_config_dict,
    }, indent=2, default=str))

    test = metrics["test"]
    print(f"[{family}/{model_name}/{dataset_name}/seed{seed}] test NDCG@10={test['ndcg@10']:.4f} "
          f"HR@10={test['hit@10']:.4f} MRR={test['mrr']:.4f} ({train_seconds:.0f}s) -> {run_dir}")
    return metrics


def cli(family):
    parser = argparse.ArgumentParser(description=f"Train a {family} recommender on a frozen split.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set epochs=2 learning_rate=0.002")
    args = parser.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    for seed in args.seeds:
        run(family, args.model, args.dataset, seed, overrides)
