"""
data/raw/のparquetから特徴量を計算してdata/processed/training_features.csvを出力する。

ネットワーク不要・sleepなし。特徴量を変更するたびにこのスクリプトだけ再実行する。

使い方:
  python build_features.py
"""

import sys
import time
from pathlib import Path

from src.constants import DATA_PROCESSED_DIR, DATA_RAW_DIR, TRAINING_DATA_PATH
from src.features import build_training_dataset_from_raw


def main() -> None:
    raw_dir = Path(DATA_RAW_DIR)
    if not raw_dir.exists() or not any(raw_dir.glob("*_event.parquet")):
        print(f"Raw data not found: {raw_dir}")
        print("Run build_raw.py first.")
        sys.exit(1)

    event_count = len(list(raw_dir.glob("*_event.parquet")))
    print(f"Building features from {event_count} rounds in {raw_dir}...")
    start = time.time()

    training_df = build_training_dataset_from_raw(str(raw_dir))

    Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    training_df.to_csv(TRAINING_DATA_PATH, index=False)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s.")
    print(f"Rows: {len(training_df)} | Columns: {len(training_df.columns)}")
    print(f"Saved: {TRAINING_DATA_PATH}")


if __name__ == "__main__":
    main()
