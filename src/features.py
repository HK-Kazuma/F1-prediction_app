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
    DRIVER_DNF_RATES_PATH,
    MAX_GAP_TO_POLE_SECONDS,
    ROLLING_AVG_WINDOW,
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
    決勝結果から finish_pos（目的変数）と GridPosition を返す。
    DNF/DSQはNaNになるためこの段階では保持し、build_feature_tableで除去する。
    columns: DriverNumber, Abbreviation, finish_pos, GridPosition
    """
    results = race_session.results[["DriverNumber", "Abbreviation", "Position", "GridPosition"]].copy()
    results["DriverNumber"] = results["DriverNumber"].astype(str)
    results = results.rename(columns={"Position": "finish_pos"})
    return results[["DriverNumber", "Abbreviation", "finish_pos", "GridPosition"]]


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

    # ペナルティでグリッドが変わった場合にその差を特徴量にする。
    # GridPosition - quali_pos > 0 はペナルティ降格、< 0 は他者のペナルティで繰り上がりを意味する。
    df["grid_penalty"] = df["GridPosition"] - df["quali_pos"]
    df = df.drop(columns=["GridPosition"])

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


def compute_driver_rolling_avg_finish(
    df: pd.DataFrame,
    window: int = ROLLING_AVG_WINDOW,
) -> pd.DataFrame:
    """
    ドライバーの直近N戦の平均フィニッシュ順位を特徴量として追加する。

    全履歴の展開平均（compute_driver_avg_finish）と違い直近フォームを重視する。
    shift(1)でそのレース自身を除外し、データリークを防ぐ。
    min_periods=1 にして window 未満でも計算する（序盤戦でNaNにならない）。
    """
    df = df.copy()
    df["driver_rolling_avg_finish"] = (
        df.groupby("driver")["finish_pos"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    return df


def compute_constructor_rolling_avg_finish(
    df: pd.DataFrame,
    window: int = ROLLING_AVG_WINDOW,
) -> pd.DataFrame:
    """
    コンストラクターの直近N戦の平均フィニッシュ順位を特徴量として追加する。

    compute_constructor_avg_finish（全履歴展開平均）の直近N戦版。
    コンストラクターの開発トレンドや直近の信頼性を捉える。
    shift(1)でそのレース自身を除外し、データリークを防ぐ。
    """
    df = df.copy()

    race_avg = (
        df.groupby(["year", "round_number", "constructor"])["finish_pos"]
        .mean()
        .reset_index(name="race_constructor_avg")
        .sort_values(["year", "round_number"])
    )

    race_avg["constructor_rolling_avg_finish"] = (
        race_avg.groupby("constructor")["race_constructor_avg"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df = df.merge(
        race_avg[["year", "round_number", "constructor", "constructor_rolling_avg_finish"]],
        on=["year", "round_number", "constructor"],
        how="left",
    )
    return df


def add_driver_rolling_avg_finish_for_prediction(
    race_df: pd.DataFrame,
    training_df: pd.DataFrame,
    window: int = ROLLING_AVG_WINDOW,
) -> pd.DataFrame:
    """
    予測時用: 学習データの直近N戦からドライバーの平均フィニッシュ順位を計算し race_df に追加する。
    未知ドライバーは NaN のまま残す（XGBoost が内部処理）。
    """
    driver_recent = (
        training_df.sort_values(["year", "round_number"])
        .groupby("driver")["finish_pos"]
        .apply(lambda x: x.tail(window).mean())
        .reset_index()
        .rename(columns={"finish_pos": "driver_rolling_avg_finish"})
    )
    return race_df.merge(driver_recent, on="driver", how="left")


def add_constructor_rolling_avg_finish_for_prediction(
    race_df: pd.DataFrame,
    training_df: pd.DataFrame,
    window: int = ROLLING_AVG_WINDOW,
) -> pd.DataFrame:
    """
    予測時用: 学習データの直近N戦からコンストラクターの平均フィニッシュ順位を計算し race_df に追加する。
    未知コンストラクターは NaN のまま残す（XGBoost が内部処理）。
    """
    race_avg = (
        training_df.groupby(["year", "round_number", "constructor"])["finish_pos"]
        .mean()
        .reset_index(name="race_constructor_avg")
        .sort_values(["year", "round_number"])
    )
    constructor_recent = (
        race_avg.groupby("constructor")["race_constructor_avg"]
        .apply(lambda x: x.tail(window).mean())
        .reset_index()
        .rename(columns={"race_constructor_avg": "constructor_rolling_avg_finish"})
    )
    return race_df.merge(constructor_recent, on="constructor", how="left")


def compute_driver_avg_finish(df: pd.DataFrame) -> pd.DataFrame:
    """
    ドライバーの過去レース平均フィニッシュ順位を特徴量として追加する。

    constructor_avg_finish と同じ設計で、ドライバー単位で集計する。
    shift(1) + expanding でそのレース自身を除外してデータリークを防ぐ。
    開幕戦など履歴がない場合は NaN（XGBoost が内部処理）。
    """
    df = df.copy()
    df["driver_avg_finish"] = (
        df.groupby("driver")["finish_pos"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    return df


def add_driver_avg_finish_for_prediction(
    race_df: pd.DataFrame,
    training_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    予測時用: 学習データ全体からドライバーの平均フィニッシュ順位を計算し
    race_df に追加する。
    """
    driver_form = (
        training_df.groupby("driver")["finish_pos"]
        .mean()
        .reset_index()
        .rename(columns={"finish_pos": "driver_avg_finish"})
    )
    return race_df.merge(driver_form, on="driver", how="left")


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

    df["grid_penalty"] = df["GridPosition"] - df["quali_pos"]
    df = df.drop(columns=["GridPosition"])

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
    results = race_results[["DriverNumber", "Abbreviation", "Position", "GridPosition"]].copy()
    results["DriverNumber"] = results["DriverNumber"].astype(str)
    return results.rename(columns={"Position": "finish_pos"})[
        ["DriverNumber", "Abbreviation", "finish_pos", "GridPosition"]
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
    combined = compute_constructor_avg_finish(combined)
    return combined


# ---------- FP3 Long Run Pace ----------

# FP3でlong runとみなすスティントの最小ラップ数。
# 3周未満はアタックラップや短いセットアップ走行と判断してlong runから除外する。
# 2018-2020年はIsAccurateの記録が不完全でスティントが短く見えるため小さめに設定する。
_FP3_MIN_LONG_RUN_LAPS = 3


def extract_fp3_long_run_pace(fp3_laps: pd.DataFrame) -> pd.DataFrame:
    """
    FP3のlong run pace（レースペース指標）をドライバーごとに返す。

    long run = スティント内で5周以上の連続走行。
    チームがFP3でレースシミュレーションを行う部分を捉える特徴量。

    処理フロー:
      1. 正確に計測されたラップのみ残す (IsAccurate=True)
      2. ピットイン/アウトラップを除外（通常より大幅に遅い）
      3. イエローフラグ・SC下のラップを除外 (TrackStatus != '1')
      4. スティントごとに周回数を集計し、_FP3_MIN_LONG_RUN_LAPS周未満は除外
      5. 各ドライバーのlong runラップの中央値をペースとして採用
      6. 最速ドライバーからのギャップ（秒）に変換（絶対タイムはサーキット依存なので相対化）

    Returns: DataFrame with columns [Driver, fp3_long_run_pace]
             fp3_long_run_pace = ドライバーの中央long runラップ - セッション最速中央ラップ（秒）
    """
    laps = fp3_laps.copy()

    if "LapTime" not in laps.columns or laps.empty:
        return pd.DataFrame(columns=["Driver", "fp3_long_run_pace"])

    # IsAccurate=Falseのラップ（計測ミス・中断）は除外するが、
    # TrueとNaN（古いシーズンは記録なし）は残す。
    # 2018-2020年はIsAccurate=Trueの割合が極端に低いため True のみに絞ると
    # サンプル数が不足してしまう。
    if "IsAccurate" in laps.columns:
        laps = laps[laps["IsAccurate"] != False]

    if "PitInTime" in laps.columns:
        laps = laps[laps["PitInTime"].isna()]
    if "PitOutTime" in laps.columns:
        laps = laps[laps["PitOutTime"].isna()]

    if "TrackStatus" in laps.columns:
        laps = laps[laps["TrackStatus"].astype(str) == "1"]

    laps = laps[laps["LapTime"].notna() & (laps["LapTime"] > 0)]

    if laps.empty:
        return pd.DataFrame(columns=["Driver", "fp3_long_run_pace"])

    # Stint列がなければLapNumberの連続性からスティントを推定する
    if "Stint" not in laps.columns or laps["Stint"].isna().all():
        laps = laps.sort_values(["Driver", "LapNumber"])
        laps["Stint"] = (
            laps.groupby("Driver")["LapNumber"]
            .transform(lambda x: (x.diff().fillna(1) != 1).cumsum())
        )

    stint_len = (
        laps.groupby(["Driver", "Stint"])["LapTime"]
        .count()
        .reset_index(name="stint_laps")
    )
    long_stints = stint_len[stint_len["stint_laps"] >= _FP3_MIN_LONG_RUN_LAPS][["Driver", "Stint"]]

    if long_stints.empty:
        # long runが一件もない場合（雨天中断等）は全accurate lapsで代替する
        pace_df = (
            laps.groupby("Driver")["LapTime"]
            .median()
            .reset_index(name="fp3_pace_seconds")
        )
    else:
        long_run_laps = laps.merge(long_stints, on=["Driver", "Stint"])
        pace_df = (
            long_run_laps.groupby("Driver")["LapTime"]
            .median()
            .reset_index(name="fp3_pace_seconds")
        )

    if pace_df.empty:
        return pd.DataFrame(columns=["Driver", "fp3_long_run_pace"])

    best_pace = pace_df["fp3_pace_seconds"].min()
    pace_df["fp3_long_run_pace"] = pace_df["fp3_pace_seconds"] - best_pace
    return pace_df[["Driver", "fp3_long_run_pace"]]


def build_training_dataset_from_raw_v2(raw_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """
    Phase 2 用学習データ構築。Phase 1 の特徴量に加えて FP3 long run pace を追加する。

    FP3データが存在しないラウンドは fp3_long_run_pace = NaN。
    XGBoostは内部でNaNを欠損として処理するため、FP3データが少なくても学習できる。
    """
    raw_path = Path(raw_dir)
    event_files = sorted(raw_path.glob("*_event.parquet"))

    if not event_files:
        raise FileNotFoundError(
            "data/raw/ にparquetが見つかりません。先に build_raw.py を実行してください。"
        )

    feature_tables = []
    for event_file in event_files:
        prefix = str(event_file)[: -len("_event.parquet")]
        race_results_path = Path(f"{prefix}_race_results.parquet")
        quali_results_path = Path(f"{prefix}_quali_results.parquet")
        weather_path = Path(f"{prefix}_race_weather.parquet")
        fp3_laps_path = Path(f"{prefix}_fp3_laps.parquet")

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

            # FP3 long run pace を追加（存在するラウンドのみ）
            if fp3_laps_path.exists():
                fp3_laps = pd.read_parquet(fp3_laps_path)
                fp3_pace = extract_fp3_long_run_pace(fp3_laps)
                # FP3のDriverコードは予選結果のAbbreviationと同一のため直接マージできる
                table = table.merge(
                    fp3_pace.rename(columns={"Driver": "driver"}),
                    on="driver",
                    how="left",
                )
            else:
                table["fp3_long_run_pace"] = np.nan

            if not table.empty:
                feature_tables.append(table)
        except Exception as error:
            print(f"[SKIP] Feature build failed for {event_file.name}: {error}")

    if not feature_tables:
        raise ValueError("有効なセッションデータが1件も得られませんでした。")

    combined = pd.concat(feature_tables, ignore_index=True)
    combined = combined.sort_values(["year", "round_number"]).reset_index(drop=True)
    combined = compute_constructor_avg_finish(combined)
    combined = compute_driver_rolling_avg_finish(combined)
    combined = compute_constructor_rolling_avg_finish(combined)
    return combined


# ---------- ドライバーDNF率 ----------

def _load_all_dnf_records(raw_dir: str) -> pd.DataFrame:
    """
    全race_resultsから (year, round_number, driver, is_dnf) を収集する。
    DNFドライバーは学習データから除外されているためraw parquetから直接読む。

    完走扱いのStatus: "Finished" / "Lapped" / "+N Laps"
    それ以外（Engine, Accident, Gearbox 等）はすべてDNFとして扱う。
    Position.isna() は DNS/Withdrew など極少数しか拾えないため使わない。
    """
    CLASSIFIED = {"Finished", "Lapped"}

    records = []
    for parquet in sorted(Path(raw_dir).glob("*_race_results.parquet")):
        parts = parquet.stem.split("_")
        year, round_num = int(parts[0]), int(parts[1])
        race_df = pd.read_parquet(parquet)[["Abbreviation", "Status"]]
        race_df["year"] = year
        race_df["round_number"] = round_num
        is_classified = (
            race_df["Status"].isin(CLASSIFIED)
            | race_df["Status"].str.match(r"^\+\d", na=False)
        )
        race_df["is_dnf"] = (~is_classified).astype(int)
        records.append(race_df[["year", "round_number", "Abbreviation", "is_dnf"]])

    all_records = pd.concat(records, ignore_index=True)
    return (
        all_records.rename(columns={"Abbreviation": "driver"})
        .sort_values(["year", "round_number"])
        .reset_index(drop=True)
    )


def compute_driver_dnf_rate(df: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    """
    ドライバーの過去レースのDNF率（0.0〜1.0）を特徴量として追加する。

    データリーク防止のため shift(1) + expanding でそのレース自身を含めない。
    初戦など履歴がない場合は NaN（XGBoost が内部処理）。
    """
    all_dnf = _load_all_dnf_records(raw_dir)
    all_dnf["driver_dnf_rate"] = (
        all_dnf.groupby("driver")["is_dnf"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    dnf_rates = all_dnf[["year", "round_number", "driver", "driver_dnf_rate"]]
    return df.merge(dnf_rates, on=["year", "round_number", "driver"], how="left")


def build_driver_dnf_rate_snapshot(raw_dir: str) -> pd.DataFrame:
    """
    予測時用: 全履歴からドライバーの累積DNF率を計算して返す。
    build_features.py が呼び出してCSVに保存し、app.py が予測時に読む。
    """
    all_dnf = _load_all_dnf_records(raw_dir)
    return (
        all_dnf.groupby("driver")["is_dnf"]
        .mean()
        .reset_index()
        .rename(columns={"is_dnf": "driver_dnf_rate"})
    )


def add_driver_dnf_rate_for_prediction(
    race_df: pd.DataFrame,
    driver_dnf_rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    予測時用: driver_dnf_rates（build_driver_dnf_rate_snapshot の出力）を
    race_df に付与する。未知ドライバーは NaN のまま残す（XGBoost が内部処理）。
    """
    return race_df.merge(driver_dnf_rates, on="driver", how="left")
