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
    FETCH_SLEEP_SECONDS_FP3,
    TRAINING_END_YEAR,
    TRAINING_START_YEAR,
)
from src.fetch import (
    configure_fastf1_cache,
    get_round_numbers_for_year,
    load_fp3_session,
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


def fp3_raw_exists(year: int, round_number: int) -> bool:
    """FP3のlapsファイルが存在するか確認する。"""
    prefix = Path(DATA_RAW_DIR) / f"{year}_{round_number:02d}"
    return Path(f"{prefix}_fp3_laps.parquet").exists()


def save_fp3_raw(year: int, round_number: int, fp3_session) -> None:
    """FP3のlapsをparquetで保存する。"""
    raw_dir = Path(DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = raw_dir / f"{year}_{round_number:02d}"
    _prepare_for_parquet(fp3_session.laps).to_parquet(
        f"{prefix}_fp3_laps.parquet", index=False
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


def _run_race_quali_fetch(args) -> None:
    """Race + Qualifying parquet を取得して保存するメインループ。"""
    sleep_between = 0 if args.fast else FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS
    sleep_after = 0 if args.fast else FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS

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


def _run_fp3_fetch(args) -> None:
    """FP3 laps parquet を取得して保存するメインループ（Phase 2用）。

    race/qualiが揃っているラウンドのみを対象にする。
    FP3はlaps=Trueで取得するためAPIコール数が多い。
    FETCH_SLEEP_SECONDS_FP3（デフォルト180s）で制御する。
    """
    sleep_after = 0 if args.fast else FETCH_SLEEP_SECONDS_FP3

    total_saved = 0
    total_skipped = 0

    for year in range(args.start_year, args.end_year + 1):
        round_numbers = get_round_numbers_for_year(year)
        print(f"\n--- FP3 {year} ({len(round_numbers)} rounds) ---")

        for round_number in round_numbers:
            if not round_raw_exists(year, round_number):
                print(f"  [SKIP] {year} Round {round_number:02d} (race/quali not yet fetched)")
                continue

            if fp3_raw_exists(year, round_number):
                print(f"  [SKIP] {year} Round {round_number:02d} (FP3 already saved)")
                total_skipped += 1
                continue

            try:
                fp3 = load_fp3_session(year, round_number)
                time.sleep(sleep_after)
                save_fp3_raw(year, round_number, fp3)
                print(f"  [OK]   {year} Round {round_number:02d}")
                total_saved += 1
            except RateLimitExceededError:
                print(f"\n[RATE LIMIT] Stopping. Saved {total_saved} FP3 rounds so far.")
                print("Re-run with --fp3 to continue (already-saved rounds will be skipped).")
                return
            except Exception as e:
                print(f"  [FAIL] {year} Round {round_number:02d}: {e}")

    print(f"\nDone. FP3 saved: {total_saved} | Skipped: {total_skipped}")
    print("Next step: python build_features_v2.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch F1 sessions and save to data/raw/")
    parser.add_argument("--start-year", type=int, default=TRAINING_START_YEAR)
    parser.add_argument("--end-year", type=int, default=TRAINING_END_YEAR)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip sleep between sessions. Safe to use when FastF1 cache is already warm.",
    )
    parser.add_argument(
        "--fp3",
        action="store_true",
        help="Fetch FP3 sessions only (for Phase 2 long run pace feature). "
             "Requires race/quali parquets to already exist.",
    )
    args = parser.parse_args()

    configure_fastf1_cache()

    if args.fp3:
        _run_fp3_fetch(args)
    else:
        _run_race_quali_fetch(args)


if __name__ == "__main__":
    main()
