"""
Phase 2 評価指標。
src/evaluate.py の全指標に加えて Spearman 順位相関を追加する。

Spearman 相関は「順位の全体的な一致度」を測るのに MAE より適した指標で、
特に20人全員の順位一致度（チャンピオン争いに限らない精度）を評価するのに使う。
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.constants import TOP_N_POSITIONS
from src.evaluate import (
    compute_baseline_mae,
    compute_baseline_top_n_hit_rate,
    compute_mae,
    compute_top_n_hit_rate,
    format_evaluation_report_for_display,
)

__all__ = [
    "compute_mae",
    "compute_top_n_hit_rate",
    "compute_baseline_mae",
    "compute_baseline_top_n_hit_rate",
    "compute_spearman_correlation",
    "build_evaluation_report",
    "format_evaluation_report_for_display",
]


def compute_spearman_correlation(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    """
    予測スコアと実際の順位の Spearman 順位相関係数を返す（-1.0〜1.0）。
    1.0 = 完全一致、0.0 = 無相関、-1.0 = 完全逆順。
    MAE が「ずれ幅」を測るのに対して、こちらは「順序の一致度」を測る。
    """
    corr, _ = spearmanr(np.array(y_true), y_pred)
    return float(corr)


def build_evaluation_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    quali_positions: pd.Series,
) -> dict:
    """
    Phase 2 の全評価指標をまとめた辞書を返す。
    Phase 1 の指標に加えて Spearman 相関を含む。
    """
    baseline_pred = quali_positions.to_numpy()
    return {
        "model_mae": compute_mae(y_true, y_pred),
        "baseline_mae": compute_baseline_mae(quali_positions, y_true),
        "model_top5_hit_rate": compute_top_n_hit_rate(y_true, y_pred),
        "baseline_top5_hit_rate": compute_baseline_top_n_hit_rate(quali_positions, y_true),
        "model_spearman": compute_spearman_correlation(y_true, y_pred),
        "baseline_spearman": compute_spearman_correlation(y_true, baseline_pred),
    }


def format_evaluation_report_for_display(report: dict) -> pd.DataFrame:
    """Phase 2 評価レポートを Streamlit テーブル用 DataFrame に変換する。"""
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
        {
            "指標": "Spearman 順位相関",
            "モデル": f"{report['model_spearman']:.3f}",
            "ベースライン（予選順位）": f"{report['baseline_spearman']:.3f}",
        },
    ]
    return pd.DataFrame(rows)
