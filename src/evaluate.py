"""
評価指標の計算。
MAEだけでは「チャンピオン争いを当てられるか」が分からないため
Top-N的中率を併用し、ベースライン（予選順位そのまま）との比較で意味を持たせる。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.constants import TOP_N_POSITIONS


def compute_mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """平均絶対誤差（MAE）を返す。予測順位と実際の順位の平均ずれ幅。"""
    return mean_absolute_error(y_true, y_pred)


def compute_top_n_hit_rate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    n: int = TOP_N_POSITIONS,
) -> float:
    """
    実際の上位N人を予測の上位N人で何割当てられたかを返す（0.0〜1.0）。
    F1の目標は「チャンピオン候補を当てること」なので中位〜下位の精度より
    上位N人の的中率の方が実用的な指標になる。
    """
    actual_top_n_indices = set(np.argsort(np.array(y_true))[:n])
    predicted_top_n_indices = set(np.argsort(y_pred)[:n])
    hit_count = len(actual_top_n_indices & predicted_top_n_indices)
    return hit_count / n


def compute_baseline_mae(
    quali_positions: pd.Series,
    actual_positions: pd.Series,
) -> float:
    """
    「予選順位 = 決勝順位」とみなすベースラインのMAEを返す。
    XGBoostがこれを下回れない場合、モデルは特徴量から何も学べていないことになる。
    """
    return mean_absolute_error(actual_positions, quali_positions)


def compute_baseline_top_n_hit_rate(
    quali_positions: pd.Series,
    actual_positions: pd.Series,
    n: int = TOP_N_POSITIONS,
) -> float:
    """ベースライン（予選順位をそのまま使った場合）のTop-N的中率を返す。"""
    return compute_top_n_hit_rate(actual_positions, quali_positions.to_numpy(), n)


def build_evaluation_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    quali_positions: pd.Series,
) -> dict:
    """
    モデルとベースラインの全評価指標をまとめた辞書を返す。
    Streamlitでの表示や学習ログ出力に使う。
    """
    return {
        "model_mae": compute_mae(y_true, y_pred),
        "baseline_mae": compute_baseline_mae(quali_positions, y_true),
        "model_top5_hit_rate": compute_top_n_hit_rate(y_true, y_pred),
        "baseline_top5_hit_rate": compute_baseline_top_n_hit_rate(quali_positions, y_true),
    }


def format_evaluation_report_for_display(report: dict) -> pd.DataFrame:
    """評価レポート辞書をStreamlitのテーブル表示用DataFrameに変換する。"""
    rows = [
        {
            "指標": "MAE（平均順位ずれ）",
            "モデル": f"{report['model_mae']:.2f}",
            "ベースライン（予選順位）": f"{report['baseline_mae']:.2f}",
        },
        {
            "指標": "Top-5 的中率",
            "モデル": f"{report['model_top5_hit_rate']:.1%}",
            "ベースライン（予選順位）": f"{report['baseline_top5_hit_rate']:.1%}",
        },
    ]
    return pd.DataFrame(rows)
