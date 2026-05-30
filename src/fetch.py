"""
FastF1 からのデータ取得モジュール。
キャッシュ設定・セッション取得・複数年一括取得を担当する。
"""

from pathlib import Path

import fastf1

from src.constants import (
    DATA_CACHE_DIR,
    SESSION_QUALIFYING,
    SESSION_RACE,
    TRAINING_END_YEAR,
    TRAINING_START_YEAR,
)


def configure_fastf1_cache(cache_dir: str = DATA_CACHE_DIR) -> None:
    """
    FastF1のローカルキャッシュを有効にする。
    キャッシュなしだと毎回APIリクエストが走り1レースの取得に数分かかるため、
    初回のみダウンロードしてローカルに保存する設計にしている。
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def load_race_session(year: int, round_number: int) -> fastf1.core.Session:
    """
    指定した年・ラウンドの決勝セッションをロードして返す。
    タイヤ・ピット・天候の特徴量を取るためにlapsとweatherを有効にする。
    テレメトリは重く不要なので無効にする。
    """
    session = fastf1.get_session(year, round_number, SESSION_RACE)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def load_qualifying_session(year: int, round_number: int) -> fastf1.core.Session:
    """
    指定した年・ラウンドの予選セッションをロードして返す。
    必要なのはresultsだけなのでlaps/weather/messagesは無効にして高速化する。
    """
    session = fastf1.get_session(year, round_number, SESSION_QUALIFYING)
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session


def get_round_numbers_for_year(year: int) -> list[int]:
    """その年の全ラウンド番号をFastF1のイベントスケジュールから取得して返す。"""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule["RoundNumber"].tolist()


def fetch_session_pair_for_round(year: int, round_number: int) -> dict | None:
    """
    1ラウンド分の予選・決勝セッションペアを辞書で返す。
    データ欠損や形式差異で取得できないラウンドはNoneを返す（スキップ用）。
    """
    try:
        race = load_race_session(year, round_number)
        quali = load_qualifying_session(year, round_number)
        return {"year": year, "round_number": round_number, "race": race, "quali": quali}
    except Exception as error:
        # 年によってはデータ未整備のラウンドが存在するため、エラーをログに残してスキップ
        print(f"[SKIP] year={year} round={round_number}: {error}")
        return None


def fetch_all_session_pairs_for_year(year: int) -> list[dict]:
    """1年分の全ラウンドについてセッションペアをリストで返す。取得失敗ラウンドは除外される。"""
    round_numbers = get_round_numbers_for_year(year)
    session_pairs = [fetch_session_pair_for_round(year, rn) for rn in round_numbers]
    return [pair for pair in session_pairs if pair is not None]


def fetch_all_session_pairs_for_years(
    start_year: int = TRAINING_START_YEAR,
    end_year: int = TRAINING_END_YEAR,
) -> list[dict]:
    """複数年分のセッションペアをまとめて返す。学習データ構築の起点となる関数。"""
    all_pairs = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        year_pairs = fetch_all_session_pairs_for_year(year)
        all_pairs.extend(year_pairs)
    return all_pairs
