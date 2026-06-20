"""
FIA Technical Regulations PDF → LightRAG Knowledge Graph 取り込み（Step 3）。

LightRAGは文書をチャンク分割し、GeminiでエンティティとRelationを抽出して
Knowledge Graphを構築する。構築後はクエリでその知識を引き出せる。

処理時間の目安（Gemini無料枠 15 RPM の場合）:
  100ページ規制文書 → 約50〜80チャンク → 5〜8分

取り込みは年1回でよい。RAG_STORAGE_DIR にKGが保存されるため、
2回目以降は既存のKGを再利用する（再実行しても安全）。
"""

import asyncio
from pathlib import Path

import pdfplumber
from lightrag import LightRAG, QueryParam

from src.constants import RAG_STORAGE_DIR
from src.rag.llm import GEMINI_EMBEDDING_FUNC, gemini_llm_func


def _read_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _build_rag(storage_dir: str = RAG_STORAGE_DIR) -> LightRAG:
    """LightRAGインスタンスを生成する。storage_dirにKGが蓄積される。"""
    return LightRAG(
        working_dir=storage_dir,
        llm_model_func=gemini_llm_func,
        embedding_func=GEMINI_EMBEDDING_FUNC,
    )


async def ingest_async(pdf_path: Path, storage_dir: str = RAG_STORAGE_DIR) -> None:
    """
    規制PDFをLightRAG KGに取り込む（非同期版）。

    Gemini無料枠のレート制限により大きいPDFは数分かかる。
    完了後はstorage_dirにKGが保存されるので再実行不要。
    """
    rag = _build_rag(storage_dir)
    text = _read_pdf(pdf_path)
    char_count = len(text)
    print(f"[Ingest] {pdf_path.name} ({char_count:,}文字) をKGに取り込み中...")
    print("[Ingest] Gemini無料枠（15 RPM）のため数分かかります。しばらくお待ちください...")
    await rag.ainsert(text)
    print(f"[Ingest] 完了: {pdf_path.name}")


def ingest(pdf_path: Path, storage_dir: str = RAG_STORAGE_DIR) -> None:
    """同期版ラッパー。"""
    asyncio.run(ingest_async(pdf_path, storage_dir))


async def query_async(
    question: str,
    mode: str = "hybrid",
    storage_dir: str = RAG_STORAGE_DIR,
) -> str:
    """
    KGに対してクエリを実行して回答を返す（非同期版）。

    mode:
      "hybrid" — KGと全文検索を組み合わせる（精度が高い、通常はこれを使う）
      "global" — KG全体を使う（「全体的な傾向は？」のような広い質問）
      "local"  — 局所的なサブグラフを使う（「XX条の内容は？」のような狭い質問）
    """
    rag = _build_rag(storage_dir)
    result = await rag.aquery(question, param=QueryParam(mode=mode))
    return result


def query(
    question: str,
    mode: str = "hybrid",
    storage_dir: str = RAG_STORAGE_DIR,
) -> str:
    """同期版ラッパー。"""
    return asyncio.run(query_async(question, mode, storage_dir))


def storage_exists(storage_dir: str = RAG_STORAGE_DIR) -> bool:
    """KGストレージが存在するか確認する。"""
    graph_file = Path(storage_dir) / "graph_chunk_entity_relation.graphml"
    return graph_file.exists()
