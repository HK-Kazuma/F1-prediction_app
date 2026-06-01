"""
XGBoost モデルの学習・予測・保存・評価用ユーティリティ。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.constants import (
    CV_N_SPLITS,
    MODEL_SAVE_PATH,
    POST_REGULATION_CHANGE_WEIGHT,
    PRE_REGULATION_CHANGE_WEIGHT,
    REGULATION_CHANGE_YEAR,
    TARGET_COLUMN,
    XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_MIN_CHILD_WEIGHT,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_RANDOM_STATE,
    XGBOOST_SUBSAMPLE,
)


def compute_era_sample_weights(years: pd.Series) -> np.ndarray:
    """
    2022年以前のデータに低い重みを付ける。
    グラウンドエフェクト規制で車体特性が根本的に変わったため、
    古いデータをそのまま等重みで使うと現行レギュレーションへの適合を妨げる。
    """
    return np.where(
        years < REGULATION_CHANGE_YEAR,
        PRE_REGULATION_CHANGE_WEIGHT,
        POST_REGULATION_CHANGE_WEIGHT,
    )


def build_xgboost_regressor() -> xgb.XGBRegressor:
    """
    ハイパーパラメータを固定したXGBoostモデルインスタンスを生成する。
    パラメータの意図はconstants.pyのコメントを参照。
    """
    return xgb.XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        subsample=XGBOOST_SUBSAMPLE,
        colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
        min_child_weight=XGBOOST_MIN_CHILD_WEIGHT,
        random_state=XGBOOST_RANDOM_STATE,
        objective="reg:squarederror",
        # XGBoostはNaNを「欠損」として内部で処理できる。
        # 将来レース（タイヤ・ピット情報未確定）の予測でNaN特徴量が混入しても壊れない。
        tree_method="hist",
    )


def _predict_with_dmatrix(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    DMatrix を明示的に構築して booster で予測する共通ヘルパー。

    xgb.DMatrix(DataFrame) はバージョンによって feature_names を自動セットしない場合がある。
    booster に保存された特徴量名を取り出し、同じ順序で列を並べた上で
    numpy 配列 + feature_names を明示的に DMatrix に渡すことで
    どのバージョンでも確実に特徴量名の照合が通るようにする。
    """
    booster = model.get_booster()
    feature_names = booster.feature_names  # 学習時に booster に保存された特徴量名

    if feature_names is not None:
        # 学習時の列順に揃えてから numpy に変換する。
        # 列順が違うと正しい特徴量に正しい重みが当たらないため必須。
        X_aligned = X.reindex(columns=feature_names)
        dmatrix = xgb.DMatrix(X_aligned.values, feature_names=feature_names)
    else:
        # 学習時に feature_names が保存されていない場合（DataFrame 以外で学習した場合）は
        # numpy に変換して検証なしで渡す
        dmatrix = xgb.DMatrix(X.values)

    return booster.predict(dmatrix)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: np.ndarray,
) -> xgb.XGBRegressor:
    """モデルを学習して返す。sample_weightsで旧レギュレーション時代のデータに低い重みを付ける。"""
    model = build_xgboost_regressor()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
) -> list[float]:
    """
    時系列交差検証でMAEスコアのリストを返す。
    F1データは年次の順序を持つため、通常のkfoldではなく TimeSeriesSplit を使う。
    未来情報を過去の予測に漏らさない（データリーク防止）ためのルール。
    """
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
    mae_scores = []

    for fold_train_idx, fold_val_idx in tscv.split(X):
        X_fold_train = X.iloc[fold_train_idx]
        X_fold_val = X.iloc[fold_val_idx]
        y_fold_train = y.iloc[fold_train_idx]
        y_fold_val = y.iloc[fold_val_idx]
        weights_fold_train = sample_weights[fold_train_idx]

        fold_model = train_model(X_fold_train, y_fold_train, weights_fold_train)
        y_pred = _predict_with_dmatrix(fold_model, X_fold_val)
        mae_scores.append(mean_absolute_error(y_fold_val, y_pred))

    return mae_scores


def predict_race_positions(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    特徴量テーブルから予測順位値（連続値）を返す。
    整数化は呼び出し側で行う（評価・表示それぞれで丸め方が異なる場合があるため）。
    """
    return _predict_with_dmatrix(model, X)


def get_feature_importance_table(
    model: xgb.XGBRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    """特徴量重要度をDataFrameで返す（Streamlitグラフ描画・学習目的の可視化に使う）。"""
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
    n_iter: int = 50,
    verbose: int = 1,
) -> dict:
    """
    RandomizedSearchCV + TimeSeriesSplit でハイパーパラメータを探索し、
    最良パラメータの辞書を返す。

    sample_weights は RandomizedSearchCV に渡すと fold ごとに自動スライスされるため、
    「旧レギュレーション時代の重みを下げる」設計はチューニング中も維持される。
    """
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)

    param_dist = {
        "n_estimators": randint(100, 501),       # 100〜500
        "max_depth": randint(3, 7),               # 3〜6
        "learning_rate": uniform(0.01, 0.09),    # 0.01〜0.10
        "subsample": uniform(0.6, 0.4),          # 0.6〜1.0
        "colsample_bytree": uniform(0.6, 0.4),   # 0.6〜1.0
        "min_child_weight": randint(1, 8),        # 1〜7
    }

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=XGBOOST_RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=XGBOOST_RANDOM_STATE,
        n_jobs=-1,
        verbose=verbose,
    )

    search.fit(X, y, sample_weight=sample_weights)
    return search.best_params_


def save_model(model: xgb.XGBRegressor, path: str = MODEL_SAVE_PATH) -> None:
    """
    モデルをJSONで保存する。
    JSONはXGBoostバージョン間の互換性が高く、pickleより安全。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_model(path: str = MODEL_SAVE_PATH) -> xgb.XGBRegressor:
    """
    保存済みモデルをロードして返す。
    predict は _predict_with_dmatrix 経由で呼ぶため、
    sklearn ラッパーの feature_names_in_ 問題は発生しない。
    """
    model = build_xgboost_regressor()
    model.load_model(path)
    return model
