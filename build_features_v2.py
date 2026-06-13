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
from src.features import build_training_dataset_from_raw


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

    # Phase 2 では FP3 long run pace を追加する
    # TODO: build_training_dataset_from_raw_v2() を実装してここで呼ぶ
    #       現時点では Phase 1 と同じ特徴量セットで動作確認用として使う
    training_df = build_training_dataset_from_raw(str(raw_dir))

    Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    training_df.to_csv(TRAINING_V2_DATA_PATH, index=False)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s.")
    print(f"Rows: {len(training_df)} | Columns: {len(training_df.columns)}")
    print(f"Saved: {TRAINING_V2_DATA_PATH}")

    if fp3_count == 0:
        print("\n[INFO] FP3 data not found. Run: python build_raw.py --fp3")
        print("       After FP3 fetch completes, re-run this script to include long run pace.")


if __name__ == "__main__":
    main()
