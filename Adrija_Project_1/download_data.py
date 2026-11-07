"""Download WikiText-2 into ./data/.

Usage:  python download_data.py

Output layout:
  data/wikitext2/wikitext-2-raw-v1/wiki.{train,valid,test}.raw     (untokenized text)
  data/wikitext2/wikitext-2-v1/wiki.{train,valid,test}.tokens      (pre-tokenized, <unk> already applied)

Source (pinned to a fixed commit for reproducibility):
  WikiText-2  : https://huggingface.co/datasets/Salesforce/wikitext
                (official mirror; the original S3 link used by mindchain/wikitext2 is dead)
"""

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent / "data"

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_CONFIGS = {"wikitext-2-raw-v1": "raw", "wikitext-2-v1": "tokens"}
WIKITEXT_SPLITS = {"train": "train", "validation": "valid", "test": "test"}


def download_wikitext2():
    for config, ext in WIKITEXT_CONFIGS.items():
        out_dir = DATA_DIR / "wikitext2" / config
        out_dir.mkdir(parents=True, exist_ok=True)
        for hf_split, file_split in WIKITEXT_SPLITS.items():
            parquet = hf_hub_download(
                WIKITEXT_REPO,
                f"{config}/{hf_split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=WIKITEXT_REV,
            )
            lines = pq.read_table(parquet).column("text").to_pylist()
            out_file = out_dir / f"wiki.{file_split}.{ext}"
            # Each row is one line of the original file (with its trailing "\n");
            # blank lines are stored as "".
            out_file.write_text("".join(line or "\n" for line in lines), encoding="utf-8")
            print(f"  {out_file.relative_to(DATA_DIR.parent)}  ({len(lines):,} lines)")


if __name__ == "__main__":
    print("Downloading WikiText-2 ...")
    download_wikitext2()
    print(f"Done. Data written to {DATA_DIR}")
