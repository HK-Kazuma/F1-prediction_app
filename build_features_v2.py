"""
Phase 2 特徴量ビルダー。
data/raw/のparquetからPhase 2用特徴量を計算して
data/processed/training_features_v2.csvを出力する。

Phase 1（build_features.py）との違い:
  - FP3 long run pace を特徴量に追加（fp3_laps.parquet が存在するラウンドのみ）
  - 将来的に他のPhase 2固有特徴量もここに追加する

使い方:
  python build_features_v2.py
"""

import sys
import time
from pathlib import Path

from src.constants import DATA_PROCESSED_DIR, DATA_RAW_DIR, TRAINING_V2_DATA_PATH
from src.features import build_training_dataset_from_raw_v2


def main() -> None:
    raw_dir = Path(DATA_RAW_DIR)
    if not raw_dir.exists() or not any(raw_dir.glob("*_event.parquet")):
        print(f"Raw data not found: {raw_dir}")
        print("Run build_raw.py first.")
        sys.exit(1)

    event_count = len(list(raw_dir.glob("*_event.parquet")))
    fp3_count = len(list(raw_dir.glob("*_fp3_laps.parquet")))
    print(f"Building Phase 2 features from {event_count} rounds ({fp3_count} with FP3)...")
    start = time.time()

    training_df = build_training_dataset_from_raw_v2(str(raw_dir))

    Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    training_df.to_csv(TRAINING_V2_DATA_PATH, index=False)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s.")
    print(f"Rows: {len(training_df)} | Columns: {len(training_df.columns)}")
    fp3_coverage = (training_df["fp3_long_run_pace"].notna().sum() / len(training_df) * 100)
    print(f"FP3 coverage: {fp3_coverage:.0f}% of rows have long run pace data")
    print(f"Saved: {TRAINING_V2_DATA_PATH}")


if __name__ == "__main__":
    main()
