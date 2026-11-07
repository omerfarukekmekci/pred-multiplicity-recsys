"""Write a frozen split (common/prepare_data.py) as RecBole atomic benchmark files.

RecBole gets the split as-is, so it never re-splits or re-filters the data:

    general     <dataset>.{train,valid,test}.inter with user_id, item_id, timestamp
    sequential  one row per target item, with the user's preceding items as item_id_list
                (RecBole only builds these lists itself when it does the splitting)
"""
import pandas as pd

from common.paths import processed_dir

SPLITS = ("train", "valid", "test")


def load_split(dataset):
    return {name: pd.read_csv(processed_dir(dataset) / f"{name}.tsv", sep="\t") for name in SPLITS}


def _write_general(split, out, name):
    header = "user_id:token\titem_id:token\ttimestamp:float\n"
    for part, df in split.items():
        with open(out / f"{name}.{part}.inter", "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            df.to_csv(f, sep="\t", index=False, header=False)


def _write_sequential(split, out, name, max_len):
    header = "user_id:token\titem_id_list:token_seq\titem_id:token\ttimestamp:float\n"
    files = {part: open(out / f"{name}.{part}.inter", "w", encoding="utf-8", newline="\n") for part in SPLITS}
    try:
        for f in files.values():
            f.write(header)
        eval_rows = {part: split[part].set_index("user") for part in ("valid", "test")}
        for user, hist in split["train"].groupby("user", sort=True):
            items, times = hist["item"].tolist(), hist["timestamp"].tolist()
            # Every train item after the first is a training target.
            for i in range(1, len(items)):
                prefix = " ".join(map(str, items[max(0, i - max_len):i]))
                files["train"].write(f"{user}\t{prefix}\t{items[i]}\t{times[i]}\n")
            for part in ("valid", "test"):
                row = eval_rows[part].loc[user]
                prefix = " ".join(map(str, items[-max_len:]))
                files[part].write(f"{user}\t{prefix}\t{row['item']}\t{row['timestamp']}\n")
                # the valid item is part of the history when predicting the test item
                items.append(row["item"])
    finally:
        for f in files.values():
            f.close()


def ensure_recbole_files(dataset, fmt, max_len=50):
    """Write the files once and return (data_path, recbole_dataset_name)."""
    name = dataset if fmt == "general" else f"{dataset}_seq{max_len}"
    root = processed_dir(dataset) / "recbole"
    out = root / name
    if not out.exists():
        # write into a temporary folder so an interrupted run never leaves partial files
        tmp = root / f"{name}.tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        split = load_split(dataset)
        if fmt == "general":
            _write_general(split, tmp, name)
        else:
            _write_sequential(split, tmp, name, max_len)
        tmp.rename(out)
    return root, name
