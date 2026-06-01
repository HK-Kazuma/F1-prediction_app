"""
プロジェクト全体で使う定数を一箇所にまとめる。
マジックナンバーをコード中に散らばらせると「なぜその値なのか」が追えなくなるため。
"""

# ---------- データ年代設定 ----------

# 2022年にグラウンドエフェクト規制が導入され、車の空力特性が根本的に変わった断絶点
REGULATION_CHANGE_YEAR = 2022

# FastF1 APIで安定してデータが取れる最古の年（それ以前はデータ欠損が多い）
TRAINING_START_YEAR = 2018
TRAINING_END_YEAR = 2024

# 汎化性能の最終確認用。モデルを調整するために繰り返し触ると2025年専用モデルになるため
# 使用は「最後の1回」に限定する
TEST_YEAR = 2025

# ---------- サンプルウェイト ----------

# 規制変更前のデータを等重みで使うと、廃止されたレギュレーション時代のパターンを
# モデルが学びすぎて2022年以降の予測精度が落ちる
PRE_REGULATION_CHANGE_WEIGHT = 0.3
POST_REGULATION_CHANGE_WEIGHT = 1.0

# ---------- XGBoost ハイパーパラメータ ----------

# F1データは年400〜500行と小規模。ニューラルネットと違いXGBoostは少データでも
# 安定するが、それでも過学習しやすいため木の深さ・学習率を保守的に設定する

XGBOOST_N_ESTIMATORS = 246
XGBOOST_MAX_DEPTH = 4           # 浅い木で局所パターンへの過適合を防ぐ
XGBOOST_LEARNING_RATE = 0.0298  # 小刻みに学習させて汎化性能を高める
XGBOOST_SUBSAMPLE = 0.7395      # 各ツリーで使うサンプル比率
XGBOOST_COLSAMPLE_BYTREE = 0.7304  # 各ツリーで使う特徴量比率
XGBOOST_MIN_CHILD_WEIGHT = 3    # リーフに最低3サンプル必要にして過学習を抑制
XGBOOST_RANDOM_STATE = 42

# ---------- 交差検証 ----------

# 時系列なのでTimeSeriesSplitを使う。5分割は小規模データでも安定したCV精度が得られる経験則
CV_N_SPLITS = 5

# ---------- 評価指標 ----------

# 仕様書で「Top-5的中率」と明示されているターゲット指標
TOP_N_POSITIONS = 5

# ---------- 目的変数・特徴量 ----------

TARGET_COLUMN = "finish_pos"

# FastF1が返すタイヤ種文字列（OneHotEncode対象）
TYRE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# ---------- サーキット種別 ----------

# FIA公式分類ではなく、予測モデルの特徴量として意味のある3種類に独自分類
CIRCUIT_TYPE_HIGH_SPEED = "high_speed"   # モンツァ・シルバーストン等、最高速重視
CIRCUIT_TYPE_STREET = "street"           # モナコ・バクー等、追い越し困難
CIRCUIT_TYPE_TECHNICAL = "technical"     # バーレーン・ハンガロリンク等、コーナリング重視

# FastF1の event["Country"] と一致させる必要がある
CIRCUIT_TYPE_MAP: dict[str, str] = {
    "Bahrain": CIRCUIT_TYPE_TECHNICAL,
    "Saudi Arabia": CIRCUIT_TYPE_HIGH_SPEED,
    "Australia": CIRCUIT_TYPE_STREET,
    "Japan": CIRCUIT_TYPE_TECHNICAL,
    "China": CIRCUIT_TYPE_TECHNICAL,
    "United States": CIRCUIT_TYPE_TECHNICAL,
    "Italy": CIRCUIT_TYPE_HIGH_SPEED,       # モンツァ
    "Monaco": CIRCUIT_TYPE_STREET,
    "Canada": CIRCUIT_TYPE_STREET,
    "Spain": CIRCUIT_TYPE_TECHNICAL,
    "Austria": CIRCUIT_TYPE_HIGH_SPEED,
    "Great Britain": CIRCUIT_TYPE_HIGH_SPEED,
    "Hungary": CIRCUIT_TYPE_TECHNICAL,
    "Belgium": CIRCUIT_TYPE_HIGH_SPEED,
    "Netherlands": CIRCUIT_TYPE_HIGH_SPEED,
    "Azerbaijan": CIRCUIT_TYPE_STREET,
    "Singapore": CIRCUIT_TYPE_STREET,
    "Mexico": CIRCUIT_TYPE_TECHNICAL,
    "Brazil": CIRCUIT_TYPE_TECHNICAL,
    "UAE": CIRCUIT_TYPE_TECHNICAL,           # アブダビ
    "Qatar": CIRCUIT_TYPE_HIGH_SPEED,
    "Las Vegas": CIRCUIT_TYPE_STREET,
    "Miami": CIRCUIT_TYPE_STREET,
}

# XGBoostに渡すためにカテゴリを整数にエンコード
CIRCUIT_TYPE_ENCODING: dict[str, int] = {
    CIRCUIT_TYPE_HIGH_SPEED: 0,
    CIRCUIT_TYPE_STREET: 1,
    CIRCUIT_TYPE_TECHNICAL: 2,
}

# マッピングにないサーキットのフォールバック値
CIRCUIT_TYPE_UNKNOWN_FALLBACK = CIRCUIT_TYPE_TECHNICAL

# ---------- データ制限値 ----------

# メカトラやタイム未設定ドライバーが大きな外れ値にならないよう上限を設ける
MAX_GAP_TO_POLE_SECONDS = 10.0

# レース開始直後の天候が戦略に最も影響するため、序盤のサンプルだけを代表値として使う
WEATHER_EARLY_SAMPLE_COUNT = 10

# ---------- FastF1 セッション識別子 ----------

SESSION_RACE = "R"
SESSION_QUALIFYING = "Q"

# ---------- APIレート制御 ----------

# FastF1の上限は500リクエスト/時 = 8.3コール/分。
# 1セッションのload()は内部で約7〜8サブリクエストを発行する。
# 1ラウンド（決勝+予選）≒ 10コール → 上限内に収めるには最低72秒/ラウンド必要。
# 安全マージンを取って90秒に設定する。

# 決勝ロード → スリープ → 予選ロードの間（同一ラウンド内の2セッション間）
FETCH_SLEEP_SECONDS_BETWEEN_SESSIONS = 15

# 1ラウンド完了 → 次のラウンド開始までの待機（コール数を1時間500以内に抑えるため）
FETCH_SLEEP_SECONDS_BETWEEN_ROUNDS = 75

# レート上限エラー発生時のリトライ設定。
# 上限リセットは1時間単位なので長めに待つ
FETCH_RATE_LIMIT_RETRY_WAIT_SECONDS = 120
FETCH_MAX_RETRIES = 3

# ---------- パス ----------

DATA_CACHE_DIR = "data/cache"
DATA_PROCESSED_DIR = "data/processed"
TRAINING_DATA_PATH = "data/processed/training_features.csv"
MODEL_SAVE_PATH = "data/processed/xgboost_model.json"
