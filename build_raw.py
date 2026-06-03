"""
生データをdata/raw/にparquet形式で保存するスクリプト。

【役割の分担】
  build_raw.py     : FastF1(キャッシュ or API) → data/raw/*.parquet  ← このスクリプト
  build_features.py: data/raw/*.parquet → data/processed/training_features.csv

特徴量を変更した場合はbuild_features.pyだけ再実行すれば良い（ネットワーク不要・数秒）。
このスクリプトはデータ取得が必要なときのみ実行する（初回 or 新しい年の追加時）。

【スキップ機能】
3つのparquetが揃っているラウンドはスキップする。中断後の再開も安全。

【使い方】
  python build_raw.py                         # 全期間（2018-2024）
  python build_raw.py --start-year 2024       # 新しい年だけ追加
  python build_raw.py --fast                  # FastF1キャッシュが温まっている場合にsleepをスキップ
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from src.constants import (
    DATA_RAW_DIR,
    FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS,
    FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS,
    TRAINING_END_YEAR,
    TRAINING_START_YEAR,
)
from src.fetch import (
    configure_fastf1_cache,
    get_round_numbers_for_year,
    load_qualifying_session,
    load_race_session,
)


def _prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Timedelta64列をfloat(秒)に変換してparquet互換にする。"""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
    return df


def round_raw_exists(year: int, round_number: int) -> bool:
    """race_results / quali_results / event の3ファイルが揃っているか確認する。"""
    prefix = Path(DATA_RAW_DIR) / f"{year}_{round_number:02d}"
    return all(
        Path(f"{prefix}_{s}.parquet").exists()
        for s in ["race_results", "quali_results", "event"]
    )


def save_round_raw(
    year: int,
    round_number: int,
    race_session,
    quali_session,
) -> None:
    """1ラウンド分の生データをparquetで保存する。"""
    raw_dir = Path(DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = raw_dir / f"{year}_{round_number:02d}"

    _prepare_for_parquet(race_session.results).to_parquet(
        f"{prefix}_race_results.parquet", index=False
    )
    _prepare_for_parquet(quali_session.results).to_parquet(
        f"{prefix}_quali_results.parquet", index=False
    )

    if race_session.weather_data is not None and not race_session.weather_data.empty:
        _prepare_for_parquet(race_session.weather_data).to_parquet(
            f"{prefix}_race_weather.parquet", index=False
        )

    pd.DataFrame([{
        "year": year,
        "round_number": round_number,
        "country": str(race_session.event.get("Country", "")),
        "circuit_name": str(race_session.event.get("Location", "")),
    }]).to_parquet(f"{prefix}_event.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch F1 sessions and save to data/raw/")
    parser.add_argument("--start-year", type=int, default=TRAINING_START_YEAR)
    parser.add_argument("--end-year", type=int, default=TRAINING_END_YEAR)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip sleep between sessions. Safe to use when FastF1 cache is already warm.",
    )
    args = parser.parse_args()

    sleep_between = 0 if args.fast else FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS
    sleep_after = 0 if args.fast else FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS

    configure_fastf1_cache()

    total_saved = 0
    total_skipped = 0

    for year in range(args.start_year, args.end_year + 1):
        round_numbers = get_round_numbers_for_year(year)
        print(f"\n--- {year} ({len(round_numbers)} rounds) ---")

        for round_number in round_numbers:
            if round_raw_exists(year, round_number):
                print(f"  [SKIP] {year} Round {round_number:02d}")
                total_skipped += 1
                continue

            try:
                race = load_race_session(year, round_number)
                time.sleep(sleep_between)
                quali = load_qualifying_session(year, round_number)
                time.sleep(sleep_after)
                save_round_raw(year, round_number, race, quali)
                print(f"  [OK]   {year} Round {round_number:02d}")
                total_saved += 1
            except RateLimitExceededError:
                print(f"\n[RATE LIMIT] Stopping. Saved {total_saved} rounds so far.")
                print("Re-run the script to continue (already-saved rounds will be skipped).")
                return
            except Exception as e:
                print(f"  [FAIL] {year} Round {round_number:02d}: {e}")

    print(f"\nDone. Saved: {total_saved} | Skipped (already exists): {total_skipped}")
    print("Next step: python build_features.py")


if __name__ == "__main__":
    main()
