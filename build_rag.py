"""
Phase 2 RAGパイプライン実行スクリプト。

実行できる処理:
  fetch     : FIAサイトからPDFを自動取得（年 + ラウンド指定）
  ingest    : 規制PDFをLightRAG KGに取り込む（年指定、初回のみ）
  dna       : チームDNAスコアを生成してJSONにキャッシュ（年指定）
  pirelli   : Pirelli PreviewからXGBoost特徴量を抽出（年 + ラウンド指定）
  all       : fetch → ingest → dna を順番に実行

使い方:
  # 2026年の規制PDFを取り込んでチームDNAスコアを生成する
  python build_rag.py ingest --year 2026
  python build_rag.py dna    --year 2026

  # 2025 Round 1 のPirelliスコアを取得する
  python build_rag.py pirelli --year 2025 --round 1 --event bahrain

  # FIA文書をまとめて取得する
  python build_rag.py fetch --year 2025 --round 1 --event bahrain

  # 規制取り込みからDNAスコア生成まで一括実行
  python build_rag.py all --year 2026
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .envからGEMINI_API_KEYを読み込む


def cmd_fetch(args: argparse.Namespace) -> None:
    from src.rag.fetch_fia import fetch_race_docs, fetch_regulation_pdf

    if args.round and args.event:
        print(f"[fetch] {args.year} Round {args.round} ({args.event}) の文書を取得中...")
        result = fetch_race_docs(args.year, args.round, args.event)
        print(f"  Pirelli Preview : {result['pirelli'] or '取得失敗（手動配置が必要）'}")
        print(f"  Event Notes     : {result['event_notes'] or '取得失敗（手動配置が必要）'}")
    else:
        print(f"[fetch] {args.year}年 Technical Regulations を取得中...")
        path = fetch_regulation_pdf(args.year)
        print(f"  Technical Regulations: {path or '取得失敗（手動配置が必要）'}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from src.rag.fetch_fia import fetch_regulation_pdf
    from src.rag.ingest import ingest, storage_exists

    if storage_exists():
        print(f"[ingest] KGストレージが既に存在します。再取り込みをスキップします。")
        print(f"         強制再実行する場合は data/rag_storage/ を削除してください。")
        return

    pdf_path = fetch_regulation_pdf(args.year)
    if pdf_path is None:
        print("[ingest] PDFが見つかりません。手動で配置してから再実行してください。")
        sys.exit(1)

    start = time.time()
    ingest(pdf_path)
    elapsed = time.time() - start
    print(f"[ingest] 完了（{elapsed:.0f}秒）")


def cmd_dna(args: argparse.Namespace) -> None:
    from src.rag.ingest import storage_exists
    from src.rag.team_dna import load_or_generate

    if not storage_exists():
        print("[dna] KGストレージがありません。先に ingest を実行してください。")
        print("      python build_rag.py ingest --year <year>")
        sys.exit(1)

    start = time.time()
    scores = load_or_generate(args.year)
    elapsed = time.time() - start

    print(f"\n[dna] {args.year}年 チームDNAスコア（{elapsed:.0f}秒）:")
    for team, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {team:20s} {bar} {score:.2f}")


def cmd_pirelli(args: argparse.Namespace) -> None:
    from src.rag.fetch_fia import fetch_race_docs
    from src.rag.pirelli import extract

    if not (args.round and args.event):
        print("[pirelli] --round と --event が必要です。")
        sys.exit(1)

    result = fetch_race_docs(args.year, args.round, args.event)
    pdf_path = result.get("pirelli")

    if pdf_path is None:
        print("[pirelli] Pirelli PreviewのPDFが見つかりません。")
        print("          手動で配置してから再実行してください。")
        sys.exit(1)

    print(f"[pirelli] {pdf_path} から特徴量を抽出中...")
    features = extract(pdf_path)
    print(f"\n[pirelli] 抽出結果:")
    for key, val in features.items():
        print(f"  {key}: {val}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_fetch(args)
    cmd_ingest(args)
    cmd_dna(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 RAGパイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd_name in ("fetch", "ingest", "dna", "pirelli", "all"):
        p = sub.add_parser(cmd_name)
        p.add_argument("--year", type=int, required=True, help="シーズン年（例: 2026）")
        p.add_argument("--round", type=int, default=None, help="ラウンド番号（pirelli/fetchで必要）")
        p.add_argument("--event", type=str, default=None, help="イベント名（例: bahrain）")

    args = parser.parse_args()
    dispatch = {
        "fetch": cmd_fetch,
        "ingest": cmd_ingest,
        "dna": cmd_dna,
        "pirelli": cmd_pirelli,
        "all": cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
