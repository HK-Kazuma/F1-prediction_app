"""
特徴量エンジニアリング。
FastF1のSessionオブジェクトから XGBoost に渡せる数値テーブルを作る。
"""

from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from src.constants import (
    CIRCUIT_TYPE_ENCODING,
    CIRCUIT_TYPE_MAP,
    CIRCUIT_TYPE_UNKNOWN_FALLBACK,
    DATA_RAW_DIR,
    MAX_GAP_TO_POLE_SECONDS,
    TARGET_COLUMN,
    WEATHER_EARLY_SAMPLE_COUNT,
)


# ---------- 予選特徴量 ----------

def extract_qualifying_features(quali_session: fastf1.core.Session) -> pd.DataFrame:
    """
    予選結果から quali_pos・gap_to_pole・constructor を返す。
    columns: DriverNumber, Abbreviation, quali_pos, gap_to_pole, constructor
    """
    results = quali_session.results[
        ["DriverNumber", "Abbreviation", "Position", "Q1", "Q2", "Q3", "TeamName"]
    ].copy()
    results["DriverNumber"] = results["DriverNumber"].astype(str)
    results = results.rename(columns={"Position": "quali_pos", "TeamName": "constructor"})
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
    return results[["DriverNumber", "Abbreviation", "quali_pos", "gap_to_pole", "constructor"]]


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
    """pandas Timedelta / datetime.timedelta / float(秒) を秒（float）に変換する。"""
    if isinstance(td, (int, float)):
        return float(td)
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

    # constructor はメタ列として保持する（constructor_avg_finish の計算に使用）
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
    combined = combined.sort_values(["year", "round_number"]).reset_index(drop=True)

    return compute_constructor_avg_finish(combined)


def compute_constructor_avg_finish(df: pd.DataFrame) -> pd.DataFrame:
    """
    コンストラクターの過去レース平均フィニッシュ順位を特徴量として追加する。

    各レースについて「そのレース以前」のレース結果から
    コンストラクターの平均フィニッシュ順位（2ドライバー分の平均）を計算する。
    データリーク防止のため、そのレース自身は含めない。
    開幕戦など過去データがない場合は NaN（XGBoost が内部処理）。
    """
    df = df.copy()

    # 1. レースごとのコンストラクター平均フィニッシュ（2ドライバーを平均）
    race_avg = (
        df.groupby(["year", "round_number", "constructor"])["finish_pos"]
        .mean()
        .reset_index(name="race_constructor_avg")
        .sort_values(["year", "round_number"])
    )

    # 2. コンストラクターごとに shift(1) + expanding mean で「過去全レース」の累積平均を計算
    #    shift(1) でそのレース自身を除外し、expanding で開幕からの全履歴を使う
    race_avg["constructor_avg_finish"] = (
        race_avg.groupby("constructor")["race_constructor_avg"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # 3. 元の DataFrame に merge（ドライバー単位で同じレースの値が入る）
    df = df.merge(
        race_avg[["year", "round_number", "constructor", "constructor_avg_finish"]],
        on=["year", "round_number", "constructor"],
        how="left",
    )

    return df


def add_constructor_avg_finish_for_prediction(
    race_df: pd.DataFrame,
    training_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    予測時用: 学習データ全体からコンストラクターの平均フィニッシュ順位を計算し
    race_df に追加する。

    学習データは「現時点までの全既知レース」なので、
    そのコンストラクターの最新の実力推定値として使える。
    """
    constructor_form = (
        training_df.groupby("constructor")["finish_pos"]
        .mean()
        .reset_index()
        .rename(columns={"finish_pos": "constructor_avg_finish"})
    )
    return race_df.merge(constructor_form, on="constructor", how="left")


# ---------- data/raw/ parquetからの特徴量構築（ネットワーク不要版） ----------

def build_feature_table_from_raw(
    race_results: pd.DataFrame,
    race_weather: pd.DataFrame | None,
    quali_results: pd.DataFrame,
    country: str,
    year: int,
    round_number: int,
) -> pd.DataFrame:
    """
    data/raw/ のparquetから読んだDataFrameで1レース分の特徴量テーブルを作る。
    build_feature_table_for_session のDataFrame版。
    Q1/Q2/Q3 はparquet保存時にfloat(秒)へ変換済みのため、
    _timedelta_to_seconds が float をそのまま返す。
    """
    quali_features = _extract_qualifying_features_from_df(quali_results)
    finish_positions = _extract_finish_positions_from_df(race_results)

    weather = _extract_weather_summary_from_df(race_weather)
    circuit_type_encoded = encode_circuit_type(country)

    df = finish_positions.merge(quali_features, on=["DriverNumber", "Abbreviation"], how="inner")
    df["air_temp"] = weather["air_temp"]
    df["rain"] = weather["rain"]
    df["circuit_type_encoded"] = circuit_type_encoded
    df["year"] = year
    df["round_number"] = round_number

    df = df.rename(columns={"Abbreviation": "driver"})
    df = df.drop(columns=["DriverNumber"])
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def _extract_qualifying_features_from_df(quali_results: pd.DataFrame) -> pd.DataFrame:
    results = quali_results[
        ["DriverNumber", "Abbreviation", "Position", "Q1", "Q2", "Q3", "TeamName"]
    ].copy()
    results["DriverNumber"] = results["DriverNumber"].astype(str)
    results = results.rename(columns={"Position": "quali_pos", "TeamName": "constructor"})

    excluded = results[results["quali_pos"].isna()]["Abbreviation"].tolist()
    if excluded:
        print(f"[INFO] Excluded drivers with no qualifying position: {excluded}")
    results = results.dropna(subset=["quali_pos"])
    results["quali_pos"] = results["quali_pos"].astype(int)

    pole_time = _get_pole_time_seconds(results)
    results["gap_to_pole"] = results.apply(
        lambda row: _compute_gap_to_pole_seconds(row, pole_time), axis=1
    )
    return results[["DriverNumber", "Abbreviation", "quali_pos", "gap_to_pole", "constructor"]]


def _extract_finish_positions_from_df(race_results: pd.DataFrame) -> pd.DataFrame:
    results = race_results[["DriverNumber", "Abbreviation", "Position"]].copy()
    results["DriverNumber"] = results["DriverNumber"].astype(str)
    return results.rename(columns={"Position": "finish_pos"})[
        ["DriverNumber", "Abbreviation", "finish_pos"]
    ]


def _extract_weather_summary_from_df(weather_df: pd.DataFrame | None) -> dict:
    if weather_df is None or weather_df.empty:
        return {"air_temp": np.nan, "rain": 0}
    early = weather_df.head(WEATHER_EARLY_SAMPLE_COUNT)
    return {
        "air_temp": early["AirTemp"].mean(),
        "rain": int(early["Rainfall"].any()),
    }


def build_training_dataset_from_raw(raw_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """
    data/raw/ のparquetファイルから学習データセット全体を構築する。
    ネットワーク不要・sleepなし。特徴量を変更したときに再実行する。
    """
    raw_path = Path(raw_dir)
    event_files = sorted(raw_path.glob("*_event.parquet"))

    if not event_files:
        raise FileNotFoundError(
            f"data/raw/ にparquetが見つかりません。先に build_raw.py を実行してください。"
        )

    feature_tables = []
    for event_file in event_files:
        prefix = str(event_file)[: -len("_event.parquet")]
        race_results_path = Path(f"{prefix}_race_results.parquet")
        quali_results_path = Path(f"{prefix}_quali_results.parquet")
        weather_path = Path(f"{prefix}_race_weather.parquet")

        if not race_results_path.exists() or not quali_results_path.exists():
            continue

        try:
            event = pd.read_parquet(event_file)
            year = int(event["year"].iloc[0])
            round_number = int(event["round_number"].iloc[0])
            country = str(event["country"].iloc[0])

            race_results = pd.read_parquet(race_results_path)
            quali_results = pd.read_parquet(quali_results_path)
            race_weather = pd.read_parquet(weather_path) if weather_path.exists() else None

            table = build_feature_table_from_raw(
                race_results=race_results,
                race_weather=race_weather,
                quali_results=quali_results,
                country=country,
                year=year,
                round_number=round_number,
            )
            if not table.empty:
                feature_tables.append(table)
        except Exception as error:
            print(f"[SKIP] Feature build failed for {event_file.name}: {error}")

    if not feature_tables:
        raise ValueError("有効なセッションデータが1件も得られませんでした。")

    combined = pd.concat(feature_tables, ignore_index=True)
    combined = combined.sort_values(["year", "round_number"]).reset_index(drop=True)
    return compute_constructor_avg_finish(combined)
