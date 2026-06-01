"""
ランダムサーチでXGBoostのハイパーパラメータを探索するスクリプト。

【使い方】
デフォルト（50イテレーション）:
    python tune_hyperparams.py

試行回数を増やして精度を上げる（時間がかかる）:
    python tune_hyperparams.py --n-iter 100

【結果の使い方】
スクリプト終了後に表示される最良パラメータを src/constants.py に手動でコピーする。
定数名との対応:
    n_estimators       -> XGBOOST_N_ESTIMATORS
    max_depth          -> XGBOOST_MAX_DEPTH
    learning_rate      -> XGBOOST_LEARNING_RATE
    subsample          -> XGBOOST_SUBSAMPLE
    colsample_bytree   -> XGBOOST_COLSAMPLE_BYTREE
    min_child_weight   -> XGBOOST_MIN_CHILD_WEIGHT
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.constants import TARGET_COLUMN, TRAINING_DATA_PATH
from src.model import compute_era_sample_weights, tune_hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning via RandomizedSearchCV")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of random parameter combinations to try (default: 50)",
    )
    return parser.parse_args()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = {"year", "round_number", "driver", TARGET_COLUMN}
    return [col for col in df.columns if col not in non_feature_cols]


def main() -> None:
    args = parse_args()

    data_path = Path(TRAINING_DATA_PATH)
    if not data_path.exists():
        print(f"Training data not found: {data_path}")
        print("Run build_dataset.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"Hyperparameter tuning -- n_iter={args.n_iter}")
    print("=" * 60)

    df = pd.read_csv(data_path)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    sample_weights = compute_era_sample_weights(df["year"])

    print(f"Rows: {len(df)} | Features: {len(feature_cols)}")
    print(f"Searching {args.n_iter} combinations (parallel on all CPU cores)...")
    print()

    start_time = time.time()
    best_params = tune_hyperparameters(X, y, sample_weights, n_iter=args.n_iter)
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print()
    print("Best parameters found:")
    print("-" * 60)

    constants_map = {
        "n_estimators": "XGBOOST_N_ESTIMATORS",
        "max_depth": "XGBOOST_MAX_DEPTH",
        "learning_rate": "XGBOOST_LEARNING_RATE",
        "subsample": "XGBOOST_SUBSAMPLE",
        "colsample_bytree": "XGBOOST_COLSAMPLE_BYTREE",
        "min_child_weight": "XGBOOST_MIN_CHILD_WEIGHT",
    }

    for param, value in sorted(best_params.items()):
        const_name = constants_map.get(param, param)
        if isinstance(value, float):
            print(f"  {const_name} = {value:.4f}   # {param}")
        else:
            print(f"  {const_name} = {value}   # {param}")

    print()
    print("Copy the values above into src/constants.py to apply them.")
    print("=" * 60)


if __name__ == "__main__":
    main()
