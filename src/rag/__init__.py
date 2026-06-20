# Phase 2: LightRAG + Gemini によるFIA文書処理パイプライン
#
# モジュール構成:
#   llm.py       — Gemini LLM + Embedding の LightRAGアダプター
#   fetch_fia.py — FIA公式サイトからのPDF自動取得
#   pirelli.py   — Pirelli Preview → XGBoost特徴量（Step 1）
#   ingest.py    — 規制PDF → LightRAG Knowledge Graph（Step 3）
#   team_dna.py  — KGクエリ → チームDNAスコア（Step 3）
