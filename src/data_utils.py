"""Đọc và làm sạch dữ liệu Foody (mục 2.2.2): bỏ rỗng, bỏ trùng trong tập, chống rò rỉ giữa các tập."""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "foody"
HF_URL = ("https://huggingface.co/datasets/pqthinh232/vietnamese-restaurant-review-sentiment-dataset"
          "/resolve/main/data/{split}-00000-of-00001.parquet")
SPLITS = {"train": "train", "dev": "validation", "test": "test"}
# nhãn gốc: 0 tiêu cực, 1 tích cực, 2 trung tính
LABEL_NAMES = {0: "Tiêu cực", 1: "Tích cực", 2: "Trung tính"}
LABEL_ORDER = [0, 2, 1]


def _key(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text)).lower()).strip()


def load_raw():
    out = {}
    for name, split in SPLITS.items():
        path = DATA_DIR / f"{split}.parquet"
        if not path.exists():
            import urllib.request
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HF_URL.format(split=split), path)
        out[name] = pd.read_parquet(path)[["review", "label"]]
    return out


def clean_splits(splits):
    stats = {}
    cleaned = {}
    for name, df in splits.items():
        n0 = len(df)
        df = df.dropna(subset=["review", "label"])
        df = df[df["review"].astype(str).str.strip() != ""].copy()
        df["key"] = df["review"].map(_key)
        df = df.drop_duplicates("key")
        cleaned[name] = df
        stats[f"{name}_dup_in_split"] = n0 - len(df)
    test_keys = set(cleaned["test"]["key"])
    n = len(cleaned["dev"])
    cleaned["dev"] = cleaned["dev"][~cleaned["dev"]["key"].isin(test_keys)]
    stats["dev_leak"] = n - len(cleaned["dev"])
    other = test_keys | set(cleaned["dev"]["key"])
    n = len(cleaned["train"])
    cleaned["train"] = cleaned["train"][~cleaned["train"]["key"].isin(other)]
    stats["train_leak"] = n - len(cleaned["train"])
    for name in cleaned:
        cleaned[name] = cleaned[name].drop(columns="key").reset_index(drop=True)
        cleaned[name]["label"] = cleaned[name]["label"].astype(int)
        stats[f"{name}_size"] = len(cleaned[name])
    return cleaned, stats
