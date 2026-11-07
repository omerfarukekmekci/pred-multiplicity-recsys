"""Turn the raw downloads into one frozen leave-one-out split per dataset.

Every recommender family reads the files written here, so all of them train and
evaluate on exactly the same interactions. Output in data/processed/<dataset>/:

    train.tsv, valid.tsv, test.tsv   user, item, timestamp (canonical integer ids)
    user_map.tsv, item_map.tsv       canonical id -> raw id
    user_features.tsv                demographics (ml-1m only)
    stats.json                       preprocessing choices and dataset statistics
    checksums.json                   sha256 of every file above

Usage:
    conda run -n pmrs python -m common.prepare_data --dataset ml-1m
    conda run -n pmrs python -m common.prepare_data --dataset all
"""
import argparse
import ast
import gzip
import hashlib
import json
import re

import numpy as np
import pandas as pd

from common.paths import RAW, processed_dir


def load_ml_1m():
    df = pd.read_csv(RAW / "ml-1m" / "ratings.dat", sep="::", engine="python",
                     names=["user", "item", "rating", "timestamp"], dtype=str)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    # Every rating counts as an implicit interaction.
    return df[["user", "item", "timestamp"]]


def load_amazon_video_games():
    rows = []
    with gzip.open(RAW / "amazon_video_games" / "Video_Games.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # parent_asin groups the variants (editions, platforms) of one product
            rows.append((r["user_id"], r["parent_asin"], r["timestamp"] // 1000))
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"])


# The Steam dump is Python 2 repr, one dict per line, so it is parsed with regexes.
_PY_STR = r"""u?(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")"""
_STEAM_FIELD = {f: re.compile(r"'%s': (%s)" % (f, _PY_STR)) for f in ("username", "product_id", "date")}


def _py_str(literal):
    if "\\" in literal:
        return ast.literal_eval(literal)
    return literal[literal.index(literal[-1]) + 1:-1]


def load_steam():
    rows = []
    with gzip.open(RAW / "steam" / "steam_reviews.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            m = [_STEAM_FIELD[k].search(line) for k in ("username", "product_id", "date")]
            if all(m):
                rows.append(tuple(_py_str(x.group(1)) for x in m))
    df = pd.DataFrame(rows, columns=["user", "item", "date"])
    # Reviews are dated to the day; ties keep file order (see sort_interactions).
    df["timestamp"] = pd.to_datetime(df["date"], format="%Y-%m-%d").astype("int64") // 10**9
    return df[["user", "item", "timestamp"]]


def load_goodreads_comics():
    rows = []
    path = RAW / "goodreads_comics" / "goodreads_interactions_comics_graphic.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            # Only books marked as read; shelving alone ("to-read") is not consumption.
            if '"is_read": true' not in line:
                continue
            r = json.loads(line)
            rows.append((r["user_id"], r["book_id"], r["date_added"]))
    df = pd.DataFrame(rows, columns=["user", "item", "date"])
    ts = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    df["timestamp"] = ts.astype("int64") // 10**9
    return df[["user", "item", "timestamp"]]


LOADERS = {
    "ml-1m": load_ml_1m,
    "amazon_video_games": load_amazon_video_games,
    "steam": load_steam,
    "goodreads_comics": load_goodreads_comics,
}


def sort_interactions(df):
    """Order each user's history by time; equal timestamps keep raw file order."""
    return df.sort_values(["user", "timestamp", "order"], kind="stable").reset_index(drop=True)


def k_core(df, k):
    while True:
        user_ok = df.groupby("user")["item"].transform("size") >= k
        item_ok = df.groupby("item")["user"].transform("size") >= k
        keep = user_ok & item_ok
        if keep.all():
            return df
        df = df[keep]


def canonical_ids(raw_ids):
    """Map raw ids to 0..n-1 in sorted order (numeric order when every id is numeric)."""
    uniq = pd.Series(raw_ids.unique())
    key = pd.to_numeric(uniq, errors="coerce")
    uniq = uniq.iloc[np.argsort(key.values, kind="stable")] if key.notna().all() else uniq.sort_values()
    return pd.Series(np.arange(len(uniq)), index=uniq.values)


def build_split(dataset, k):
    df = LOADERS[dataset]()
    n_raw = len(df)
    df["order"] = np.arange(len(df))
    df = sort_interactions(df)
    # Keep the first interaction of every (user, item) pair, so a test item is
    # never already in the user's history.
    df = df.drop_duplicates(["user", "item"], keep="first")
    n_dedup = len(df)
    df = k_core(df, k)

    user_ids, item_ids = canonical_ids(df["user"]), canonical_ids(df["item"])
    df = df.assign(user=df["user"].map(user_ids).values, item=df["item"].map(item_ids).values)
    df = sort_interactions(df)

    # Leave-one-out by time: last interaction -> test, second-to-last -> valid.
    from_end = df.groupby("user").cumcount(ascending=False)
    split = {"train": df[from_end >= 2], "valid": df[from_end == 1], "test": df[from_end == 0]}
    return df, split, user_ids, item_ids, {"raw_interactions": n_raw, "after_dedup": n_dedup}


def compute_stats(df, split, k, counts):
    lengths = df.groupby("user").size()
    same_ts = (df["user"].values[1:] == df["user"].values[:-1]) & \
              (df["timestamp"].values[1:] == df["timestamp"].values[:-1])
    train_items = set(split["train"]["item"])
    n_users, n_items = df["user"].nunique(), df["item"].nunique()
    return {
        "preprocessing": {
            "k_core": k,
            "dedup": "first interaction of each (user, item) pair",
            "split": "leave-one-out by time (last -> test, second-to-last -> valid)",
            "tie_break": "raw file order",
        },
        **counts,
        "users": int(n_users),
        "items": int(n_items),
        "interactions": int(len(df)),
        "density": len(df) / (n_users * n_items),
        "split_sizes": {name: int(len(part)) for name, part in split.items()},
        "history_length": {
            "mean": float(lengths.mean()),
            **{f"p{q}": float(lengths.quantile(q / 100)) for q in (5, 25, 50, 75, 95)},
            "max": int(lengths.max()),
        },
        "users_with_lt10_interactions": int((lengths < 10).sum()),
        "share_consecutive_same_timestamp": float(same_ts.sum() / max(len(df) - n_users, 1)),
        "eval_items_unseen_in_train": {
            name: int((~split[name]["item"].isin(train_items)).sum()) for name in ("valid", "test")
        },
        "time_span": [str(pd.to_datetime(df["timestamp"].min(), unit="s")),
                      str(pd.to_datetime(df["timestamp"].max(), unit="s"))],
    }


def ml_1m_user_features(user_ids):
    users = pd.read_csv(RAW / "ml-1m" / "users.dat", sep="::", engine="python",
                        names=["raw", "gender", "age", "occupation", "zip"], dtype=str)
    users = users[users["raw"].isin(user_ids.index)]
    users.insert(0, "user", users["raw"].map(user_ids))
    return users.drop(columns="raw").sort_values("user")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(dataset, k):
    out = processed_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    df, split, user_ids, item_ids, counts = build_split(dataset, k)

    for name, part in split.items():
        part[["user", "item", "timestamp"]].to_csv(out / f"{name}.tsv", sep="\t", index=False)
    pd.DataFrame({"user": user_ids.values, "raw_id": user_ids.index}).to_csv(out / "user_map.tsv", sep="\t", index=False)
    pd.DataFrame({"item": item_ids.values, "raw_id": item_ids.index}).to_csv(out / "item_map.tsv", sep="\t", index=False)
    if dataset == "ml-1m":
        ml_1m_user_features(user_ids).to_csv(out / "user_features.tsv", sep="\t", index=False)

    stats = compute_stats(df, split, k, counts)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "checksums.json")
    (out / "checksums.json").write_text(json.dumps({p.name: sha256(p) for p in files}, indent=2))
    print(f"[{dataset}] users={stats['users']} items={stats['items']} "
          f"interactions={stats['interactions']} density={stats['density']:.5f} -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=[*LOADERS, "all"])
    parser.add_argument("--k_core", type=int, default=5, help="minimum interactions per user and per item")
    args = parser.parse_args()
    if args.k_core < 3:
        parser.error("leave-one-out needs at least 3 interactions per user")
    for dataset in LOADERS if args.dataset == "all" else [args.dataset]:
        prepare(dataset, args.k_core)


if __name__ == "__main__":
    main()
