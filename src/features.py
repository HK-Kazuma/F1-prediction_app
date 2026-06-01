"""
特徴量エンジニアリング。
FastF1のSessionオブジェクトから XGBoost に渡せる数値テーブルを作る。
"""

import fastf1
import numpy as np
import pandas as pd

from src.constants import (
    CIRCUIT_TYPE_ENCODING,
    CIRCUIT_TYPE_MAP,
    CIRCUIT_TYPE_UNKNOWN_FALLBACK,
    MAX_GAP_TO_POLE_SECONDS,
    TARGET_COLUMN,
    WEATHER_EARLY_SAMPLE_COUNT,
)


# ---------- 予選特徴量 ----------

def extract_qualifying_features(quali_session: fastf1.core.Session) -> pd.DataFrame:
    """
    予選結果から quali_pos と gap_to_pole を返す。
    columns: DriverNumber, Abbreviation, quali_pos, gap_to_pole
    """
    results = quali_session.results[
        ["DriverNumber", "Abbreviation", "Position", "Q1", "Q2", "Q3"]
    ].copy()
    results["DriverNumber"] = results["DriverNumber"].astype(str)
    results = results.rename(columns={"Position": "quali_pos"})
    # 予選不出走（DNS/DSQ等）のドライバーはPositionがNaNになる。
    # quali_posが使えない行は特徴量として意味がないため除外する。
    excluded = results[results["quali_pos"].isna()]["Abbreviation"].tolist()
    if excluded:
        print(f"[INFO] Excluded drivers with no qualifying position: {excluded}")
    results = results.dropna(subset=["quali_pos"])
    results["quali_pos"] = results["quali_pos"].astype(int)

    pole_time_seconds = _get_pole_time_seconds(results)
    results["gap_to_pole"] = results.apply(
        lambda row: _compute_gap_to_pole_seconds(row, pole_time_seconds), axis=1
    )
    return results[["DriverNumber", "Abbreviation", "quali_pos", "gap_to_pole"]]


def _get_pole_time_seconds(quali_results: pd.DataFrame) -> float:
    """
    ポール（Q1位）のベストタイムを秒で返す。
    gap_to_pole の基準値なので全ドライバー共通の参照点として使う。
    """
    pole_row = quali_results[quali_results["quali_pos"] == 1].iloc[0]
    # ポール取得者は必ずQ3タイムを持つが、レッドフラッグ等の異常時に備えてフォールバックする
    for session_col in ["Q3", "Q2", "Q1"]:
        if pd.notna(pole_row[session_col]):
            return _timedelta_to_seconds(pole_row[session_col])
    raise ValueError("Pole position driver has no valid lap time in any qualifying session")


def _compute_gap_to_pole_seconds(row: pd.Series, pole_time_seconds: float) -> float:
    """
    各ドライバーのポールとのタイム差（秒）を計算する。
    Q3→Q2→Q1の順に有効タイムを探す。Q3未出走（Q1/Q2敗退）のドライバーは
    その段階のベストタイムで比較するため、ノックアウト前の実力差が反映される。
    上限を設けるのは、DNS/メカトラ等による外れ値がモデルを歪めるのを防ぐため。
    """
    for session_col in ["Q3", "Q2", "Q1"]:
        if pd.notna(row[session_col]):
            gap = _timedelta_to_seconds(row[session_col]) - pole_time_seconds
            return min(gap, MAX_GAP_TO_POLE_SECONDS)
    # タイムが一切ない（DNS等）は最大ペナルティ扱い
    return MAX_GAP_TO_POLE_SECONDS


def _timedelta_to_seconds(td) -> float:
    """pandas Timedelta または datetime.timedelta を秒（float）に変換する。"""
    return pd.Timedelta(td).total_seconds()


# ---------- 決勝結果（目的変数） ----------

def extract_finish_positions(race_session: fastf1.core.Session) -> pd.DataFrame:
    """
    決勝結果から finish_pos（目的変数）を返す。
    DNF/DSQはNaNになるためこの段階では保持し、build_feature_tableで除去する。
    columns: DriverNumber, Abbreviation, finish_pos
    """
    results = race_session.results[["DriverNumber", "Abbreviation", "Position"]].copy()
    results["DriverNumber"] = results["DriverNumber"].astype(str)
    results = results.rename(columns={"Position": "finish_pos"})
    return results[["DriverNumber", "Abbreviation", "finish_pos"]]


# ---------- 天候特徴量 ----------

def extract_weather_summary(race_session: fastf1.core.Session) -> dict:
    """
    レース開始直後の気温と雨フラグを返す。
    序盤の天候が戦略（タイヤ選択・ピットタイミング）に最も影響するため
    レース全体の平均ではなく序盤のサンプルを使う。
    """
    weather = race_session.weather_data

    if weather is None or weather.empty:
        # 天候データが取得できなかった場合は欠損扱い（XGBoostが内部でNaN処理する）
        return {"air_temp": np.nan, "rain": 0}

    early_weather = weather.head(WEATHER_EARLY_SAMPLE_COUNT)
    air_temp = early_weather["AirTemp"].mean()
    rain = int(early_weather["Rainfall"].any())
    return {"air_temp": air_temp, "rain": rain}


# ---------- サーキット種別 ----------

def encode_circuit_type(country_name: str) -> int:
    """
    国名（FastF1の event["Country"]）を circuit_type の整数コードに変換する。
    未知のサーキットはTECHNICALにフォールバックする（最も多くのサーキットが該当するため）。
    """
    circuit_type = CIRCUIT_TYPE_MAP.get(country_name, CIRCUIT_TYPE_UNKNOWN_FALLBACK)
    return CIRCUIT_TYPE_ENCODING[circuit_type]


# ---------- セッション→特徴量テーブル変換 ----------

def build_feature_table_for_session(
    race_session: fastf1.core.Session,
    quali_session: fastf1.core.Session,
    year: int,
    round_number: int,
) -> pd.DataFrame:
    """
    1レース分の予選・決勝セッションから特徴量テーブル（1行=1ドライバー）を作る。
    各特徴量グループをDriverNumberをキーにmergeして1つのテーブルに統合する。
    """
    quali_features = extract_qualifying_features(quali_session)
    finish_positions = extract_finish_positions(race_session)
    weather = extract_weather_summary(race_session)
    circuit_type_encoded = encode_circuit_type(race_session.event["Country"])

    # 決勝結果を基準にinner joinすることで、予選データがないドライバーを除外する
    df = finish_positions.merge(quali_features, on=["DriverNumber", "Abbreviation"], how="inner")

    df["air_temp"] = weather["air_temp"]
    df["rain"] = weather["rain"]
    df["circuit_type_encoded"] = circuit_type_encoded
    df["year"] = year
    df["round_number"] = round_number

    df = df.rename(columns={"Abbreviation": "driver"})
    df = df.drop(columns=["DriverNumber"])

    # DNF/DSQはfinish_posがNaN。目的変数が存在しない行は学習に使えないため除外する
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


# ---------- 複数セッション→学習データセット ----------

def build_training_dataset(session_pairs: list[dict]) -> pd.DataFrame:
    """
    複数年分のセッションペアリストから学習用データセット全体を構築して返す。
    各レースのテーブルを縦に結合する。
    """
    feature_tables = []

    for pair in session_pairs:
        try:
            table = build_feature_table_for_session(
                race_session=pair["race"],
                quali_session=pair["quali"],
                year=pair["year"],
                round_number=pair["round_number"],
            )
            if not table.empty:
                feature_tables.append(table)
        except Exception as error:
            # 1レースの失敗が全体を止めないようにスキップして継続する
            print(f"[SKIP] Feature build failed for year={pair['year']} round={pair['round_number']}: {error}")

    if not feature_tables:
        raise ValueError("有効なセッションデータが1件も得られませんでした。")

    combined = pd.concat(feature_tables, ignore_index=True)

    # 時系列順に並べる（TimeSeriesSplitのために年・ラウンドの昇順が必要）
    return combined.sort_values(["year", "round_number"]).reset_index(drop=True)
