# F1 Race Prediction App

F1決勝レースの順位をデータ分析・機械学習により予測するアプリケーション。

予選タイム・タイヤ戦略・天候・ピット回数などのレースデータをもとに XGBoost で順位を予測し、Streamlit のダッシュボードで結果を可視化する。

---

## 機能（Phase 1）

- FastF1 を使った 2018〜2024 年の F1 レースデータ取得・キャッシュ
- 予選順位・タイヤ・ピット・天候などの特徴量エンジニアリング
- XGBoost による決勝順位予測
- 時系列交差検証（TimeSeriesSplit）によるモデル評価
- ベースライン（予選順位そのまま）との精度比較
- Streamlit ダッシュボードでの予測順位表・特徴量重要度グラフ表示

---

## 動作環境

- Python 3.10 以上
- インターネット接続（初回のデータ取得時のみ）

---

## セットアップ手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/<your-username>/F1-prediction_app.git
cd F1-prediction_app
```

### 2. 仮想環境を作成・有効化

```bash
# 仮想環境を作成
python -m venv venv

# 有効化（Mac / Linux）
source venv/bin/activate

# 有効化（Windows PowerShell）
venv\Scripts\Activate.ps1
```

### 3. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

---

## データ取得

学習データは FastF1 API から取得する。  
**初回のみ実行が必要。取得後はローカルキャッシュに保存されるため2回目以降は不要。**

### 手順

#### ① 特定の年だけ試したい場合（推奨・短時間）

```bash
python build_dataset.py --start-year 2023 --end-year 2024
```

#### ② 全学習データを取得する場合（完全版）

年ごとに分けて実行することを推奨（API レート上限: 500リクエスト/時）。

```bash
python build_dataset.py --start-year 2018 --end-year 2018
python build_dataset.py --start-year 2019 --end-year 2019
python build_dataset.py --start-year 2020 --end-year 2020
python build_dataset.py --start-year 2021 --end-year 2021
python build_dataset.py --start-year 2022 --end-year 2022
python build_dataset.py --start-year 2023 --end-year 2023
python build_dataset.py --start-year 2024 --end-year 2024
```

全年のキャッシュが揃ったら CSV を一括作成する。

```bash
python build_dataset.py --start-year 2018 --end-year 2024 --force
```

#### 注意事項

- 1年あたりの取得時間：約 40〜60 分（キャッシュなしの場合）
- 途中で止めても問題なし。`Ctrl+C` で中断後、同じコマンドを再実行するとキャッシュ済みのラウンドをスキップして再開する
- キャッシュの保存先：`data/cache/`
- 生成される CSV：`data/processed/training_features.csv`

---

## アプリの起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開く。

---

## アプリの使い方

### サイドバー

| 項目 | 説明 |
|---|---|
| シーズン | 予測したいレースの年を選択 |
| ラウンド番号 | 第何戦かを入力（1〜24）|
| サーキット名・国名 | 選択したラウンドの開催地が自動表示される |

### 特徴量重要度グラフ

モデルがどの特徴量を重視しているかを棒グラフで表示する。  
「予選順位（quali_pos）」が最も重要な特徴量になることが多い。

### 予測を実行

選択したレースのデータを取得し、決勝順位を予測して表示する。  
過去のレースを選択した場合は実際の結果との比較（MAE・Top-5的中率）も表示される。

### 交差検証

時系列交差検証（TimeSeriesSplit）を実行し、モデルの汎化性能を確認する。  
ベースライン（予選順位そのまま）との比較で、モデルの有効性を評価する。

---

## 評価指標

| 指標 | 説明 | Phase 1 目標 |
|---|---|---|
| MAE（平均絶対誤差） | 予測順位と実際の順位の平均ずれ幅 | 4〜6位以内 |
| Top-5 的中率 | 上位5位以内のドライバーをどれだけ当てられるか | 40〜50% |
| ベースライン比 | 「予選順位そのまま」より精度が高いか | 上回れば合格 |

---

## ファイル構成

```
F1-prediction_app/
├── app.py                  # Streamlit ダッシュボード
├── build_dataset.py        # 学習データ収集スクリプト（初回のみ実行）
├── explore_data.py         # FastF1 データ構造の確認スクリプト
├── requirements.txt        # 依存パッケージ
├── data/
│   ├── cache/              # FastF1 キャッシュ（自動生成・git管理外）
│   └── processed/          # 学習データ CSV・モデルファイル（自動生成・git管理外）
├── src/
│   ├── constants.py        # 全定数の定義
│   ├── fetch.py            # FastF1 データ取得
│   ├── features.py         # 特徴量エンジニアリング
│   ├── model.py            # XGBoost 学習・予測・保存
│   ├── evaluate.py         # 評価指標の計算
│   └── rag/                # Phase 2 以降（LightRAG 連携）
└── notebooks/              # 分析・実験用 Jupyter Notebook
```

---

## 開発ロードマップ

| フェーズ | 内容 | 状態 |
|---|---|---|
| Phase 1 | FastF1 データ + XGBoost MVP | ✅ 完了 |
| Phase 2 | LightRAG + LLM によるチームDNAスコア統合 | 🔜 次フェーズ |
| Phase 3 | ベイズ更新 + モンテカルロシミュレーション | 🔜 将来 |

---

## 技術スタック

| カテゴリ | ライブラリ |
|---|---|
| データ取得 | FastF1, OpenF1 API |
| データ加工 | pandas, scikit-learn |
| ML 予測 | XGBoost |
| UI | Streamlit |
| グラフ描画 | Plotly |
