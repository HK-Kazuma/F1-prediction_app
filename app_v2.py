"""
Streamlit ダッシュボード — Phase 2。
Phase 1 (app.py) との違い:
  - Phase 2 モデル（MODEL_V2_SAVE_PATH）を使用
  - 評価指標に Spearman 順位相関を追加
  - 将来: FP3 long run pace 可視化、Monte Carlo シミュレーション

Phase 1 を完全に保持したまま独立して動かせる設計にしている。
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.constants import (
    COUNTRY_ABBR,
    DATA_PROCESSED_DIR,
    MODEL_V2_SAVE_PATH,
    TARGET_COLUMN,
    TEST_YEAR,
    TRAINING_V2_DATA_PATH,
    TRAINING_END_YEAR,
    TRAINING_START_YEAR,
)
from src.features import (
    add_constructor_avg_finish_for_prediction,
    add_constructor_rolling_avg_finish_for_prediction,
    add_driver_rolling_avg_finish_for_prediction,
    build_feature_table_for_session,
)
from src.fetch import (
    configure_fastf1_cache,
    get_round_schedule,
    load_qualifying_session,
    load_race_session,
)
from src.phase2.evaluate import (
    build_evaluation_report,
    format_evaluation_report_for_display,
)
from src.phase2.model import (
    compute_era_sample_weights,
    get_feature_importance_table,
    load_model,
    predict_race_positions,
    run_cross_validation,
    save_model,
    train_model,
)

# ---------- UI定数 ----------

PAGE_TITLE = "F1 Race Prediction — Phase 2"
F1_RED = "#E8002D"
IMPORTANCE_TOP_N = 15
PREDICTION_TOP_N = 10

# ---------- ヘルパー関数 ----------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """学習に使う特徴量列名だけを返す。"""
    non_feature_cols = {"year", "round_number", "driver", "constructor", TARGET_COLUMN}
    return [col for col in df.columns if col not in non_feature_cols]


@st.cache_data(show_spinner=False)
def load_round_schedule(year: int) -> list[dict]:
    """サイドバーのラウンド選択に使うスケジュール情報をキャッシュして返す。"""
    configure_fastf1_cache()
    return get_round_schedule(year)


@st.cache_data(show_spinner=False)
def load_training_data_from_csv() -> pd.DataFrame:
    return pd.read_csv(TRAINING_V2_DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_or_train_model(training_data_path: str) -> object:
    """
    保存済みモデルがあればロード、なければ学習して保存する。
    特徴量セットが変わっていたら自動再学習する。
    """
    df = pd.read_csv(training_data_path)
    feature_cols = get_feature_columns(df)

    if Path(MODEL_V2_SAVE_PATH).exists():
        candidate = load_model()
        booster_names = candidate.get_booster().feature_names
        if booster_names is not None and list(booster_names) == feature_cols:
            return candidate
        Path(MODEL_V2_SAVE_PATH).unlink()

    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    weights = compute_era_sample_weights(df["year"])

    model = train_model(X, y, weights)
    save_model(model)
    return model


# ---------- 描画関数 ----------

def render_feature_importance_section(model, feature_cols: list[str]) -> None:
    st.header("特徴量重要度")
    importance_df = get_feature_importance_table(model, feature_cols)

    fig = px.bar(
        importance_df.head(IMPORTANCE_TOP_N),
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[F1_RED],
        title=f"Top {IMPORTANCE_TOP_N} Feature Importances (XGBoost gain)",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_table(predictions_df: pd.DataFrame) -> None:
    """予測順位表を描画する。実際の結果がある場合は結果列も表示する。"""
    has_actual = (
        TARGET_COLUMN in predictions_df.columns
        and predictions_df[TARGET_COLUMN].notna().all()
    )

    cols = ["predicted_rank", "driver", "quali_pos"]
    rename = {
        "predicted_rank": "予測順位",
        "driver": "ドライバー",
        "quali_pos": "グリッド",
    }

    if has_actual:
        cols.append(TARGET_COLUMN)
        rename[TARGET_COLUMN] = "実際の順位"

    cols.append("predicted_pos_raw")
    rename["predicted_pos_raw"] = "スコア（低いほど上位）"

    display_df = (
        predictions_df[cols]
        .rename(columns=rename)
        .head(PREDICTION_TOP_N)
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_evaluation_section(
    y_true: pd.Series,
    y_pred,
    quali_positions: pd.Series,
) -> None:
    """Phase 2 評価指標（MAE + Top-5 + Spearman）を表示する。"""
    st.subheader("モデル vs ベースライン（予選順位そのまま）")
    report = build_evaluation_report(y_true, y_pred, quali_positions)
    report_df = format_evaluation_report_for_display(report)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    mae_better  = report["model_mae"]           < report["baseline_mae"]
    top5_better = report["model_top5_hit_rate"] > report["baseline_top5_hit_rate"]
    top5_equal  = report["model_top5_hit_rate"] == report["baseline_top5_hit_rate"]
    mae_diff    = report["baseline_mae"]         - report["model_mae"]
    top5_diff   = report["model_top5_hit_rate"]  - report["baseline_top5_hit_rate"]

    if mae_better and (top5_better or top5_equal):
        st.success(
            f"MAE をベースライン比 {mae_diff:.2f} 改善 ✓  "
            f"／  Top-5 は {'同率' if top5_equal else f'{top5_diff:.0%} 改善'}"
        )
    elif mae_better and not top5_better:
        st.warning(
            f"MAE はベースライン比 {mae_diff:.2f} 改善 ✓  "
            f"／  Top-5 はベースラインに {abs(top5_diff):.0%} 負け ⚠️"
        )
    elif not mae_better and (top5_better or top5_equal):
        st.warning(
            f"MAE はベースラインより {abs(mae_diff):.2f} 悪化 ⚠️  "
            f"／  Top-5 は {'同率' if top5_equal else f'{top5_diff:.0%} 改善'} ✓"
        )
    else:
        st.error("MAE・Top-5 ともにベースライン以下 ❌  特徴量・パラメータを見直してください。")


def render_cv_section(train_df: pd.DataFrame, feature_cols: list[str]) -> None:
    st.header("交差検証（Time Series CV）")
    st.caption(
        "未来データのリークを防ぐため TimeSeriesSplit を使用。"
        f"直近データを検証セットにしながら {st.session_state.get('cv_n_splits', 5)} 回評価する。"
    )

    if st.button("交差検証を実行"):
        X = train_df[feature_cols]
        y = train_df[TARGET_COLUMN]
        weights = compute_era_sample_weights(train_df["year"])

        with st.spinner("交差検証中..."):
            mae_scores = run_cross_validation(X, y, weights)

        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(mae_scores))],
            "MAE": [round(s, 3) for s in mae_scores],
        })
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        mean_mae = sum(mae_scores) / len(mae_scores)
        st.metric("平均 CV MAE", f"{mean_mae:.3f}", help="値が小さいほど予測精度が高い")


# ---------- メイン ----------

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🏎", layout="wide")
    st.title("🏎 F1 Race Prediction — Phase 2")

    st.sidebar.header("設定")
    selected_year = st.sidebar.selectbox("シーズン", options=[TEST_YEAR], index=0)

    round_schedule = load_round_schedule(selected_year)
    round_labels = [
        f"Round {r['round_number']:02d}  {COUNTRY_ABBR.get(r['country'], '???')}  {r['location']}"
        for r in round_schedule
    ]
    selected_label = st.sidebar.selectbox("ラウンド", round_labels)
    selected_idx = round_labels.index(selected_label)
    selected_round = round_schedule[selected_idx]["round_number"]
    event_info = {
        "circuit_name": round_schedule[selected_idx]["location"],
        "country": round_schedule[selected_idx]["country"],
    }

    st.sidebar.divider()
    st.sidebar.caption(
        f"学習データ: {TRAINING_START_YEAR}〜{TRAINING_END_YEAR}\n"
        f"テストデータ: {TEST_YEAR}"
    )

    if not Path(TRAINING_V2_DATA_PATH).exists():
        st.error("Phase 2 学習データが見つかりません。")
        st.info(
            "**以下のコマンドでデータを準備してください:**\n\n"
            "```\n"
            "python build_raw.py --fp3          # FP3データ取得（初回のみ・約8〜10時間）\n"
            "python build_features_v2.py         # 特徴量計算（数秒）\n"
            "```"
        )
        st.stop()

    with st.spinner("学習データを読み込み中..."):
        train_df = load_training_data_from_csv()

    feature_cols = get_feature_columns(train_df)

    with st.spinner("モデルを読み込み中..."):
        model = load_or_train_model(TRAINING_V2_DATA_PATH)

    render_feature_importance_section(model, feature_cols)

    st.divider()

    st.header(
        f"レース予測: {selected_year} Round {selected_round} "
        f"— {event_info['circuit_name']} ({event_info['country']})"
    )

    if st.button("予測を実行", type="primary"):
        with st.spinner("データ取得中..."):
            configure_fastf1_cache()
            try:
                race_session = load_race_session(selected_year, selected_round)
                quali_session = load_qualifying_session(selected_year, selected_round)
            except Exception as error:
                st.error(f"セッションデータの取得に失敗しました: {error}")
                return

        try:
            race_df = build_feature_table_for_session(
                race_session, quali_session, selected_year, selected_round
            )
            train_df = load_training_data_from_csv()
            race_df = add_constructor_avg_finish_for_prediction(race_df, train_df)
            race_df = add_driver_rolling_avg_finish_for_prediction(race_df, train_df)
            race_df = add_constructor_rolling_avg_finish_for_prediction(race_df, train_df)
        except Exception as error:
            st.error(f"特徴量テーブルの構築に失敗しました: {error}")
            return

        X_race = race_df.reindex(columns=feature_cols, fill_value=0)
        predicted_positions = predict_race_positions(model, X_race)

        race_df["predicted_pos_raw"] = predicted_positions
        race_df["predicted_rank"] = (
            race_df["predicted_pos_raw"].rank(method="first").astype(int)
        )
        race_df = race_df.sort_values("predicted_rank").reset_index(drop=True)

        render_prediction_table(race_df)

        has_actual_results = (
            TARGET_COLUMN in race_df.columns
            and race_df[TARGET_COLUMN].notna().all()
        )
        if has_actual_results:
            st.divider()
            render_evaluation_section(
                y_true=race_df[TARGET_COLUMN],
                y_pred=race_df["predicted_pos_raw"].values,
                quali_positions=race_df["quali_pos"],
            )

    st.divider()
    render_cv_section(train_df, feature_cols)


if __name__ == "__main__":
    main()
