"""
学習データセットをオフラインで構築するスクリプト。

【なぜappと分離するか】
FastF1のAPIレート上限は500リクエスト/時。
2018〜2024年の全データ取得には数百回のAPIコールが必要で、
Streamlit起動時に自動実行すると上限超過エラーが起きる。
このスクリプトを手動実行してCSVを作っておき、
app.pyはそのCSVを読むだけにする。

【APIレート制御】
年ごとに分割して取得し、年間終了後に待機時間を設けている。
これにより、短時間の大量リクエストを防ぐ。

【使い方】
全年を取得（時間がかかる）:
    python build_dataset.py

特定の年だけ試す:
    python build_dataset.py --start-year 2023 --end-year 2024

特定の年1個だけ:
    python build_dataset.py --start-year 2023 --end-year 2023
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.constants import (
    DATA_PROCESSED_DIR,
    FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS,
    FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS,
    TRAINING_DATA_PATH,
    TRAINING_END_YEAR,
    TRAINING_START_YEAR,
)
from src.features import build_training_dataset
from src.fetch import (
    configure_fastf1_cache,
    fetch_all_session_pairs_for_years,
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Build F1 training dataset from FastF1 API")
    parser.add_argument(
        "--start-year",
        type=int,
        default=TRAINING_START_YEAR,
        help=f"First year to fetch (default: {TRAINING_START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=TRAINING_END_YEAR,
        help=f"Last year to fetch (default: {TRAINING_END_YEAR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV even if it already exists",
    )
    return parser.parse_args()


def print_progress(year: int, round_number: int, success: bool) -> None:
    """ラウンドごとの取得状況をターミナルに出力する進捗コールバック。"""
    status = "OK" if success else "SKIP"
    print(f"  [{status}] {year} Round {round_number:02d}")


def main() -> None:
    args = parse_args()

    output_path = Path(TRAINING_DATA_PATH)
    if output_path.exists() and not args.force:
        print(f"Dataset already exists: {output_path}")
        print("Use --force to rebuild from scratch.")
        sys.exit(0)

    # 予想時間を計算して表示（キャッシュ済みラウンドは実際にはスキップされるため最大値）
    # 1年あたりのラウンド数は変動するが平均22で計算
    AVERAGE_ROUNDS_PER_YEAR = 22
    total_years = args.end_year - args.start_year + 1
    estimated_rounds = total_years * AVERAGE_ROUNDS_PER_YEAR
    estimated_minutes = estimated_rounds * (FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS + FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS) / 60

    print("=" * 70)
    print(f"Building training dataset: {args.start_year} - {args.end_year}")
    print(f"API rate control: {FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS}s (between sessions)")
    print(f"                  {FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS}s (between rounds)")
    print(f"Estimated time (no cache): ~{estimated_minutes:.0f} min")
    print("Cached rounds will be skipped and take only seconds.")
    print("=" * 70)

    configure_fastf1_cache()
    start_time = time.time()

    all_pairs = []
    total_years = args.end_year - args.start_year + 1

    # 年ごとに分割して取得する。各年の終了後に30秒の待機を入れ、
    # 短時間の大量リクエストを防ぐ。
    for idx, year in enumerate(range(args.start_year, args.end_year + 1), start=1):
        print(f"\n[{idx}/{total_years}] Fetching {year}...")
        year_pairs = fetch_all_session_pairs_for_years(
            start_year=year,
            end_year=year,
            on_round_fetched=print_progress,
        )
        all_pairs.extend(year_pairs)

        # 次の年があれば待機
        if year < args.end_year:
            wait_seconds = 30
            print(f"Waiting {wait_seconds}s before next year (API rate control)...")
            time.sleep(wait_seconds)

    print("\nBuilding feature table...")
    training_df = build_training_dataset(all_pairs)

    Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_path, index=False)

    elapsed_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print(f"Done. Rows: {len(training_df)} | Columns: {len(training_df.columns)}")
    print(f"Saved to: {output_path}")
    print(f"Elapsed: {elapsed_minutes:.1f} min")
    print("=" * 70)
    print("\nYou can now launch the app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
