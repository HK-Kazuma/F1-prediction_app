"""
Phase 2 モデルユーティリティ。
src/model.py と同一の学習・予測ロジックを持つが、
保存先パスが MODEL_V2_SAVE_PATH になっている。

Phase 2 固有の拡張（アンサンブル・確率的予測など）はここに追記する。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.constants import MODEL_V2_SAVE_PATH
from src.model import (
    _predict_with_dmatrix,
    build_xgboost_regressor,
    compute_era_sample_weights,
    get_feature_importance_table,
    run_cross_validation,
    train_model,
)

__all__ = [
    "compute_era_sample_weights",
    "get_feature_importance_table",
    "run_cross_validation",
    "train_model",
    "predict_race_positions",
    "save_model",
    "load_model",
]


def predict_race_positions(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """特徴量テーブルから予測順位値（連続値）を返す。"""
    return _predict_with_dmatrix(model, X)


def save_model(model: xgb.XGBRegressor, path: str = MODEL_V2_SAVE_PATH) -> None:
    """Phase 2 モデルを JSON で保存する（デフォルト先: MODEL_V2_SAVE_PATH）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_model(path: str = MODEL_V2_SAVE_PATH) -> xgb.XGBRegressor:
    """保存済み Phase 2 モデルをロードして返す。"""
    model = build_xgboost_regressor()
    model.load_model(path)
    return model
