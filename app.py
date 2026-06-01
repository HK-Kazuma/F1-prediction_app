"""
Streamlit ダッシュボード — Phase 1 MVP。
予測順位表・特徴量重要度グラフ・評価指標の3ビューを提供する。
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.constants import (
    DATA_PROCESSED_DIR,
    MODEL_SAVE_PATH,
    TARGET_COLUMN,
    TEST_YEAR,
    TRAINING_DATA_PATH,
    TRAINING_END_YEAR,
    TRAINING_START_YEAR,
)
from src.evaluate import build_evaluation_report, format_evaluation_report_for_display
from src.features import build_feature_table_for_session
from src.fetch import (
    configure_fastf1_cache,
    get_event_info,
    get_round_numbers_for_year,
    load_qualifying_session,
    load_race_session,
)
from src.model import (
    compute_era_sample_weights,
    get_feature_importance_table,
    load_model,
    predict_race_positions,
    run_cross_validation,
    save_model,
    train_model,
)

# ---------- UI定数 ----------

PAGE_TITLE = "F1 Race Prediction — Phase 1"
F1_RED = "#E8002D"          # F1公式カラー。ブランドの一貫性のために使う
IMPORTANCE_TOP_N = 15       # 重要度グラフに表示する特徴量数。多すぎると読みにくくなる
PREDICTION_TOP_N = 10       # 予測結果テーブルの表示行数

# ---------- ヘルパー関数 ----------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    学習に使う特徴量列名だけを返す。
    メタ列（year, round, driver）と目的変数を除外する必要があるため、
    除外リストで明示的に管理する。
    """
    non_feature_cols = {"year", "round_number", "driver", TARGET_COLUMN}
    return [col for col in df.columns if col not in non_feature_cols]


@st.cache_data(show_spinner=False)
def load_training_data_from_csv() -> pd.DataFrame:
    """
    CSVから学習データを読み込む。
    データ収集はアプリと分離して build_dataset.py で行う。
    Streamlit起動時にAPI大量呼び出しが走るとレート上限に引っかかるため、
    app.pyはCSVを読むだけに徹する設計にしている。
    """
    return pd.read_csv(TRAINING_DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_or_train_model(training_data_path: str) -> object:
    """
    保存済みモデルがあればロード、なければ学習して保存する。
    st.cache_resourceでモデルオブジェクトをセッション間で共有する。
    training_data_pathをキーにすることでデータが変わったときにキャッシュが無効化される。
    """
    if Path(MODEL_SAVE_PATH).exists():
        return load_model()

    df = pd.read_csv(training_data_path)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    weights = compute_era_sample_weights(df["year"])

    model = train_model(X, y, weights)
    save_model(model)
    return model


# ---------- 描画関数 ----------

def render_feature_importance_section(model, feature_cols: list[str]) -> None:
    """特徴量重要度グラフを描画する。モデルがどの情報を最も重視しているか学習目的で確認できる。"""
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
    """予測順位表を描画する。"""
    display_df = (
        predictions_df[["predicted_rank", "driver", "quali_pos", "predicted_pos_raw"]]
        .rename(columns={
            "predicted_rank": "予測順位",
            "driver": "ドライバー",
            "quali_pos": "グリッド",
            "predicted_pos_raw": "スコア（低いほど上位）",
        })
        .head(PREDICTION_TOP_N)
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_evaluation_section(
    y_true: pd.Series,
    y_pred,
    quali_positions: pd.Series,
) -> None:
    """モデル vs ベースラインの評価指標を表示する。"""
    st.subheader("モデル vs ベースライン（予選順位そのまま）")
    report = build_evaluation_report(y_true, y_pred, quali_positions)
    report_df = format_evaluation_report_for_display(report)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # MAE・Top-5 の両方を見て総合判定する。
    # MAEだけ見ると Top-5 でベースラインに負けていても「改善」と表示されてしまうため。
    mae_better   = report["model_mae"]             < report["baseline_mae"]
    top5_better  = report["model_top5_hit_rate"]   > report["baseline_top5_hit_rate"]
    top5_equal   = report["model_top5_hit_rate"]  == report["baseline_top5_hit_rate"]
    mae_diff     = report["baseline_mae"]          - report["model_mae"]
    top5_diff    = report["model_top5_hit_rate"]   - report["baseline_top5_hit_rate"]

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
        st.error(
            f"MAE・Top-5 ともにベースライン以下 ❌  "
            f"特徴量・パラメータを見直してください。"
        )


def render_cv_section(train_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """時系列交差検証を実行して結果を表示する。"""
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
    st.title("🏎 F1 Race Prediction — Phase 1 MVP")

    # --- サイドバー ---
    st.sidebar.header("設定")
    selected_year = st.sidebar.selectbox(
        "シーズン",
        options=[TEST_YEAR],
        index=0,
    )
    selected_round = st.sidebar.number_input(
        "ラウンド番号", min_value=1, max_value=24, value=1, step=1
    )

    # ラウンド番号だけでは開催地が分からないため、サーキット名・国名を取得して表示する
    configure_fastf1_cache()
    event_info = get_event_info(selected_year, int(selected_round))
    st.sidebar.info(
        f"🏁 **{event_info['circuit_name']}**\n\n"
        f"🌍 {event_info['country']}"
    )

    st.sidebar.divider()
    st.sidebar.caption(
        f"学習データ: {TRAINING_START_YEAR}〜{TRAINING_END_YEAR}\n"
        f"テストデータ: {TEST_YEAR}"
    )

    # --- 学習データの存在確認 ---
    # データ収集は build_dataset.py で行う。CSVがなければ手順を案内して停止する。
    if not Path(TRAINING_DATA_PATH).exists():
        st.error("学習データが見つかりません。")
        st.info(
            "**最初に以下のコマンドでデータを収集してください（一度だけ実行）:**\n\n"
            "```\n"
            "python build_dataset.py\n"
            "```\n\n"
            "特定の年だけ試したい場合:\n\n"
            "```\n"
            "python build_dataset.py --start-year 2023 --end-year 2024\n"
            "```"
        )
        st.stop()

    # --- 学習データとモデルの読み込み ---
    with st.spinner("学習データを読み込み中..."):
        train_df = load_training_data_from_csv()

    feature_cols = get_feature_columns(train_df)

    with st.spinner("モデルを読み込み中..."):
        model = load_or_train_model(TRAINING_DATA_PATH)

    # --- 特徴量重要度 ---
    render_feature_importance_section(model, feature_cols)

    st.divider()

    # --- レース予測 ---
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
        except Exception as error:
            st.error(f"特徴量テーブルの構築に失敗しました: {error}")
            return

        # 学習時と同じ列順に揃える。未知の特徴量は0で補完する
        X_race = race_df.reindex(columns=feature_cols, fill_value=0)
        predicted_positions = predict_race_positions(model, X_race)

        race_df["predicted_pos_raw"] = predicted_positions
        race_df["predicted_rank"] = (
            race_df["predicted_pos_raw"].rank(method="first").astype(int)
        )
        race_df = race_df.sort_values("predicted_rank").reset_index(drop=True)

        render_prediction_table(race_df)

        # 実際の結果がある（過去レース）場合は評価も表示する
        has_actual_results = (
            TARGET_COLUMN in race_df.columns
            and race_df[TARGET_COLUMN].notna().all()
        )
        if has_actual_results:
            st.divider()
            render_evaluation_section(
                y_true=race_df[TARGET_COLUMN],
                y_pred=predicted_positions,
                quali_positions=race_df["quali_pos"],
            )

    st.divider()

    # --- 交差検証 ---
    render_cv_section(train_df, feature_cols)


if __name__ == "__main__":
    main()
