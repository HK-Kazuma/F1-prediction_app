"""
FastF1 からのデータ取得モジュール。
キャッシュ設定・セッション取得・複数年一括取得を担当する。
"""

import time
from pathlib import Path

import fastf1
from fastf1.exceptions import RateLimitExceededError

from src.constants import (
    DATA_CACHE_DIR,
    FETCH_MAX_RETRIES,
    FETCH_RATE_LIMIT_RETRY_WAIT_SECONDS,
    FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS,
    FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS,
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


def _load_session_with_retry(
    year: int,
    round_number: int,
    session_type: str,
    load_kwargs: dict,
) -> fastf1.core.Session:
    """
    セッションをロードする。RateLimitExceededError が発生したときだけリトライする。
    それ以外のエラー（データ欠損・形式不正等）は即座に再送出して呼び出し側でスキップさせる。
    """
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(**load_kwargs)
            return session
        except RateLimitExceededError:
            if attempt == FETCH_MAX_RETRIES:
                raise
            print(
                f"[RATE LIMIT] year={year} round={round_number} {session_type}. "
                f"Waiting {FETCH_RATE_LIMIT_RETRY_WAIT_SECONDS}s before retry "
                f"({attempt}/{FETCH_MAX_RETRIES})..."
            )
            time.sleep(FETCH_RATE_LIMIT_RETRY_WAIT_SECONDS)


def load_race_session(year: int, round_number: int) -> fastf1.core.Session:
    """
    指定した年・ラウンドの決勝セッションをロードして返す。
    タイヤ・ピット・天候の特徴量を取るためにlapsとweatherを有効にする。
    テレメトリは重く不要なので無効にする。
    """
    return _load_session_with_retry(
        year, round_number, SESSION_RACE,
        {"laps": False, "telemetry": False, "weather": True, "messages": False},
    )


def load_qualifying_session(year: int, round_number: int) -> fastf1.core.Session:
    """
    指定した年・ラウンドの予選セッションをロードして返す。
    必要なのはresultsだけなのでlaps/weather/messagesは無効にして高速化する。
    """
    return _load_session_with_retry(
        year, round_number, SESSION_QUALIFYING,
        {"laps": False, "telemetry": False, "weather": False, "messages": False},
    )


def get_round_numbers_for_year(year: int) -> list[int]:
    """その年の全ラウンド番号をFastF1のイベントスケジュールから取得して返す。"""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule["RoundNumber"].tolist()


def get_event_info(year: int, round_number: int) -> dict:
    """
    指定した年・ラウンドのサーキット名と国名を返す。
    ラウンド番号だけではどこで開催されるか分からないため、
    サイドバーやヘッダーの表示に使う。
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    event = schedule[schedule["RoundNumber"] == round_number]

    if event.empty:
        return {"circuit_name": "Unknown", "country": "Unknown"}

    row = event.iloc[0]
    return {
        "circuit_name": str(row.get("Location", "Unknown")),
        "country": str(row.get("Country", "Unknown")),
    }


def fetch_session_pair_for_round(year: int, round_number: int) -> dict | None:
    """
    1ラウンド分の予選・決勝セッションペアを辞書で返す。
    データ欠損や形式差異で取得できないラウンドはNoneを返す（スキップ用）。

    スリープの構造:
      決勝ロード → BETWEEN_SESSIONS待機 → 予選ロード → BETWEEN_ROUNDS待機
    ラウンド内とラウンド間で別の待機時間を設けることで、
    1時間500コールの上限を守りながら無駄な待機を最小化する。
    """
    try:
        race = load_race_session(year, round_number)
        # 同一ラウンド内の2セッション間は短めに待機
        time.sleep(FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS)

        quali = load_qualifying_session(year, round_number)
        # ラウンド完了後は長めに待機してレート上限を守る
        time.sleep(FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS)

        return {"year": year, "round_number": round_number, "race": race, "quali": quali}
    except RateLimitExceededError:
        # リトライ上限を超えたら呼び出し元（build_dataset.py）に伝えて処理を止める
        raise
    except Exception as error:
        # データ未整備・フォーマット差異のラウンドはスキップして続行する
        print(f"[SKIP] year={year} round={round_number}: {error}")
        return None


def fetch_all_session_pairs_for_year(
    year: int,
    on_round_fetched: callable = None,
) -> list[dict]:
    """
    1年分の全ラウンドについてセッションペアをリストで返す。取得失敗ラウンドは除外される。
    on_round_fetched はラウンド取得完了のたびに呼ばれるコールバック（進捗表示用）。
    """
    round_numbers = get_round_numbers_for_year(year)
    session_pairs = []

    for round_number in round_numbers:
        pair = fetch_session_pair_for_round(year, round_number)
        if pair is not None:
            session_pairs.append(pair)
        if on_round_fetched is not None:
            on_round_fetched(year, round_number, success=pair is not None)

    return session_pairs


def fetch_all_session_pairs_for_years(
    start_year: int = TRAINING_START_YEAR,
    end_year: int = TRAINING_END_YEAR,
    on_round_fetched: callable = None,
) -> list[dict]:
    """複数年分のセッションペアをまとめて返す。学習データ構築の起点となる関数。"""
    all_pairs = []
    for year in range(start_year, end_year + 1):
        print(f"--- Fetching {year} ---")
        year_pairs = fetch_all_session_pairs_for_year(year, on_round_fetched=on_round_fetched)
        all_pairs.extend(year_pairs)
    return all_pairs
