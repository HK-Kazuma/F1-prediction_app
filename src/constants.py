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

# 国名 → 3文字略称マッピング（FastF1の event["Country"] と一致させる）
COUNTRY_ABBR: dict[str, str] = {
    "Bahrain":       "BHR",
    "Saudi Arabia":  "KSA",
    "Australia":     "AUS",
    "Japan":         "JPN",
    "China":         "CHN",
    "United States": "USA",
    "Italy":         "ITA",
    "Monaco":        "MON",
    "Canada":        "CAN",
    "Spain":         "ESP",
    "Austria":       "AUT",
    "Great Britain": "GBR",
    "Hungary":       "HUN",
    "Belgium":       "BEL",
    "Netherlands":   "NED",
    "Azerbaijan":    "AZE",
    "Singapore":     "SGP",
    "Mexico":        "MEX",
    "Brazil":        "BRA",
    "UAE":           "UAE",
    "Qatar":         "QAT",
    "Las Vegas":     "USA",
    "Miami":         "USA",
}

# 国名 → 国旗絵文字マッピング（FastF1の event["Country"] と一致させる）
COUNTRY_FLAG: dict[str, str] = {
    "Bahrain":       "🇧🇭",
    "Saudi Arabia":  "🇸🇦",
    "Australia":     "🇦🇺",
    "Japan":         "🇯🇵",
    "China":         "🇨🇳",
    "United States": "🇺🇸",
    "Italy":         "🇮🇹",
    "Monaco":        "🇲🇨",
    "Canada":        "🇨🇦",
    "Spain":         "🇪🇸",
    "Austria":       "🇦🇹",
    "Great Britain": "🇬🇧",
    "Hungary":       "🇭🇺",
    "Belgium":       "🇧🇪",
    "Netherlands":   "🇳🇱",
    "Azerbaijan":    "🇦🇿",
    "Singapore":     "🇸🇬",
    "Mexico":        "🇲🇽",
    "Brazil":        "🇧🇷",
    "UAE":           "🇦🇪",
    "Qatar":         "🇶🇦",
    "Las Vegas":     "🇺🇸",
    "Miami":         "🇺🇸",
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
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"

# ---------- Phase 1 ----------
TRAINING_DATA_PATH = "data/processed/training_features.csv"
DRIVER_DNF_RATES_PATH = "data/processed/driver_dnf_rates.csv"
MODEL_SAVE_PATH = "data/processed/xgboost_model.json"

# ---------- Phase 2 ----------
TRAINING_V2_DATA_PATH = "data/processed/training_features_v2.csv"
MODEL_V2_SAVE_PATH = "data/processed/xgboost_model_v2.json"

# ---------- Phase 2 / RAG ----------

# LightRAG がKnowledge Graphのインデックスを保存するディレクトリ
RAG_STORAGE_DIR = "data/rag_storage"

# FIA Technical Regulations PDF の保存先 (年ごと)
# 例: data/regulations/2026/technical_regulations.pdf
FIA_REGULATIONS_DIR = "data/regulations"

# レース週末資料 (Pirelli Preview / Race Director Notes) の保存先
# 例: data/race_docs/2025/round_01_bahrain/pirelli_preview.pdf
FIA_RACE_DOCS_DIR = "data/race_docs"

# Pirelli特徴量スコアのキャッシュ (年×ラウンド → スコア)
PIRELLI_FEATURES_PATH = "data/processed/pirelli_features.csv"

# チームDNAスコアのキャッシュ (年ごとにJSONで保存)
# 実際のファイル名は team_dna_scores_{year}.json になる
TEAM_DNA_SCORES_PATH = "data/processed/team_dna_scores.json"

# Gemini 無料枠は15 RPM。LLM呼び出し間のスリープ秒数
# 4.5秒 × 10チーム = 45秒でチームDNAスコアが揃う
GEMINI_RATE_LIMIT_SLEEP_SECONDS = 4.5

# ドライバー・コンストラクターのローリング平均フィニッシュのウィンドウサイズ。
# 全履歴の展開平均より直近フォームを重視しつつ、
# ウィンドウが小さすぎると1レースの異常値に過敏になるため5戦に設定する。
ROLLING_AVG_WINDOW = 5

# FP3は laps=True で取得するため1セッションあたりのサブリクエストが多い。
# 500コール/時の上限を守るため通常より長めの待機を設ける。
FETCH_SLEEP_SECONDS_FP3 = 180
