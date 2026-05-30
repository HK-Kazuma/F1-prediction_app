"""
FastF1データの構造をターミナルで確認するスクリプト。
実際のレースデータを取得して、どんなカラム・データがあるか可視化する。
"""

import fastf1
from pathlib import Path

from src.constants import DATA_CACHE_DIR
from src.fetch import load_race_session, load_qualifying_session

# キャッシュ設定
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(DATA_CACHE_DIR)

print("=" * 80)
print("FastF1 データ構造の確認")
print("=" * 80)

# サンプル: 2023年シンガポール（複雑なレースで有名）
year, round_num = 2023, 17

print(f"\n[1] Race Session: {year} Round {round_num} (Singapore)")
race = load_race_session(year, round_num)

print("\n[Race.results - Final standings & qualifying times]")
print(f"  Columns: {race.results.columns.tolist()}")
print("\n  Top 3 drivers:")
print(race.results[["DriverNumber", "Abbreviation", "Position", "Q1", "Q3"]].head(3).to_string())

print("\n[Race.laps - Each driver's laps]")
print(f"  Columns: {race.laps.columns.tolist()}")
print("\n  Red Bull drivers lap 1-3:")
redBull_laps = race.laps[race.laps["Team"] == "Red Bull Racing"].head(3)
print(
    redBull_laps[["Driver", "LapNumber", "Compound", "TyreLife", "PitInTime", "LapTime"]].to_string()
)

print("\n[Race.weather_data - Temperature & rain]")
print(f"  Columns: {race.weather_data.columns.tolist()}")
print("\n  First 5 samples:")
print(race.weather_data[["Time", "AirTemp", "Rainfall", "TrackTemp"]].head(5).to_string())

# 予選
print("\n" + "=" * 80)
print(f"[2] Qualifying Session: {year} Round {round_num}")
quali = load_qualifying_session(year, round_num)

print("\n[Quali.results - Qualifying standings]")
print("\n  Top 5 drivers:")
print(quali.results[["DriverNumber", "Abbreviation", "Position", "Q1", "Q2", "Q3"]].head(5).to_string())

print("\n" + "=" * 80)
print("DATA STATISTICS")
print("=" * 80)
print(f"Finishers (Race): {race.results[race.results['Position'].notna()].shape[0]}")
print(f"Total laps logged: {race.laps.shape[0]} (all drivers combined)")
print(f"Weather samples: {race.weather_data.shape[0]}")

print("\n[FEATURES EXTRACTED FROM THIS DATA]")
print("  - quali_pos: Qualifying position")
print("  - gap_to_pole: Gap to pole time (seconds)")
print("  - tyre_compound: Tire compound at start (SOFT/MEDIUM/HARD/WET)")
print("  - tyre_age: Tire age at start (laps used)")
print("  - pit_count: Number of pit stops")
print("  - last_pit_lap: Final pit stop lap number")
print("  - air_temp: Air temperature (Celsius)")
print("  - rain: Rain flag (0/1)")
print("  - finish_pos: Final finishing position (TARGET VARIABLE)")
