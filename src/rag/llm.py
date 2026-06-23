"""
Gemini LLM + Embedding wrappers for LightRAG.

LightRAGはデフォルトでOpenAIを使う設計だが、カスタム関数を渡せる。
このファイルはそのインターフェースに合わせたGeminiアダプター。

llm_model_func  : テキスト生成 (entity抽出・クエリ回答)
embedding_func  : テキストのベクトル化 (KGの類似検索用)

どちらも Gemini 無料枠で動く:
  - LLM      : gemini-1.5-flash (15 RPM / 1500 RPD)
  - Embedding: text-embedding-004 (100 RPM / 1500 RPD)
"""

import asyncio
import os
from typing import Any

import google.generativeai as genai
from lightrag.utils import EmbeddingFunc

_GEMINI_LLM_MODEL = "gemini-2.0-flash"
_GEMINI_EMBED_MODEL = "models/text-embedding-004"
_EMBED_DIM = 768       # text-embedding-004 の出力次元数
_EMBED_MAX_TOKENS = 8192


def _configure() -> None:
    """環境変数からAPIキーを設定する。呼び出し毎に設定しても副作用はない。"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY が設定されていません。"
            ".env ファイルを作成して GEMINI_API_KEY=<your_key> を設定してください。"
        )
    genai.configure(api_key=api_key)


async def gemini_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] = [],
    **kwargs: Any,
) -> str:
    """
    LightRAGの llm_model_func インターフェース実装。
    LightRAGは内部でこの関数をエンティティ抽出・クエリ回答に使う。

    google-generativeai は同期APIのみのため、
    run_in_executor でイベントループをブロックしないようにする。
    """
    _configure()
    model = genai.GenerativeModel(
        model_name=_GEMINI_LLM_MODEL,
        system_instruction=system_prompt or "You are a helpful assistant.",
    )
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: model.generate_content(prompt),
    )
    return response.text


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    テキストのリストをGemini Embeddingでベクトル化して返す。
    無料枠 100 RPM に対して逐次処理するため大量テキストは時間がかかる。
    """
    _configure()
    loop = asyncio.get_event_loop()
    results = []
    for text in texts:
        response = await loop.run_in_executor(
            None,
            lambda t=text: genai.embed_content(
                model=_GEMINI_EMBED_MODEL,
                content=t,
                task_type="retrieval_document",
            ),
        )
        results.append(response["embedding"])
    return results


# LightRAGの embedding_func に渡すオブジェクト
# embedding_dim はtext-embedding-004の固定次元数
GEMINI_EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=_EMBED_DIM,
    max_token_size=_EMBED_MAX_TOKENS,
    func=_embed_texts,
)
