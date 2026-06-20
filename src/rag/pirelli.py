"""
Pirelli Preview PDF からレース特徴量を抽出するモジュール（Step 1）。

LightRAGは使わない。
Pirelli Previewは2〜3ページなのでPDFテキストをそのままGeminiに渡せば十分。

抽出する特徴量（XGBoostに渡す数値）:
  pirelli_compound_min   : 使用コンパウンドのうち最軟コンパウンドのC番号 (1〜6)
  pirelli_compound_max   : 使用コンパウンドのうち最硬コンパウンドのC番号 (1〜6)
  pirelli_predicted_stops: 予測ピット回数 (1〜3)
  pirelli_degradation    : タイヤ劣化リスク (0.0〜1.0、1.0=最大)
  pirelli_track_evolution: トラック進化度 (0.0〜1.0、1.0=ゴム乗り最大)

C番号について:
  C1(最硬)〜C6(最軟)。毎レース3種選択される。
  例: C2/C3/C4 → compound_min=2, compound_max=4
  硬いコンパウンドほど安定・遅い。軟らかいほど速いが劣化しやすい。
"""

import asyncio
import json
import re
from pathlib import Path

import pdfplumber

from src.rag.llm import gemini_llm_func

_SYSTEM = (
    "You are an F1 data extraction assistant. "
    "Extract structured data from Pirelli tire preview documents. "
    "Respond with valid JSON only. No explanation or markdown."
)

_PROMPT = """\
Extract the following from this Pirelli F1 tire preview document:

1. compound_min: The softest compound's C-number selected for this race (integer 1-6)
   C1=hardest/most durable, C5 or C6=softest/fastest but most fragile
2. compound_max: The hardest compound's C-number selected (integer 1-6)
3. predicted_stops: Most likely number of pit stops recommended (integer 1-3)
4. degradation: Tire degradation severity score (float 0.0-1.0)
   0.0=very low degradation, 1.0=extreme degradation expected
5. track_evolution: Expected rubber buildup on track during the race (float 0.0-1.0)
   0.0=track stays green/slippery, 1.0=heavy rubber laid down (lap times improve a lot)

Respond ONLY with this JSON:
{{
  "pirelli_compound_min": <int>,
  "pirelli_compound_max": <int>,
  "pirelli_predicted_stops": <int>,
  "pirelli_degradation": <float>,
  "pirelli_track_evolution": <float>
}}

Document:
{text}
"""

# LLM呼び出し失敗時のフォールバック値（中立的な値）
FALLBACK_FEATURES: dict[str, float | int] = {
    "pirelli_compound_min": 3,
    "pirelli_compound_max": 5,
    "pirelli_predicted_stops": 2,
    "pirelli_degradation": 0.5,
    "pirelli_track_evolution": 0.5,
}

# Pirelli Previewは2〜3ページなので4000文字に収まるが、念のため上限を設ける
_TEXT_CHAR_LIMIT = 5000


def _read_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _parse_response(response: str) -> dict:
    """LLMのレスポンスからJSONを抽出する。余計なテキストが混じっても対応。"""
    match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
    if not match:
        raise ValueError(f"JSON not found in response: {response[:300]}")
    return json.loads(match.group())


def _validate(raw: dict) -> dict[str, float | int]:
    """型変換と値域チェック。不正値はフォールバック値で補完する。"""
    fb = FALLBACK_FEATURES
    return {
        "pirelli_compound_min": int(max(1, min(6, raw.get("pirelli_compound_min", fb["pirelli_compound_min"])))),
        "pirelli_compound_max": int(max(1, min(6, raw.get("pirelli_compound_max", fb["pirelli_compound_max"])))),
        "pirelli_predicted_stops": int(max(1, min(3, raw.get("pirelli_predicted_stops", fb["pirelli_predicted_stops"])))),
        "pirelli_degradation": float(max(0.0, min(1.0, raw.get("pirelli_degradation", fb["pirelli_degradation"])))),
        "pirelli_track_evolution": float(max(0.0, min(1.0, raw.get("pirelli_track_evolution", fb["pirelli_track_evolution"])))),
    }


async def extract_async(pdf_path: Path) -> dict[str, float | int]:
    """
    Pirelli Preview PDFから特徴量を抽出する（非同期版）。
    LLMの呼び出しに失敗した場合はフォールバック値を返す（例外は投げない）。
    """
    try:
        text = _read_pdf(pdf_path)
    except Exception as exc:
        print(f"[Pirelli] PDF読み込み失敗 ({pdf_path.name}): {exc}")
        return dict(FALLBACK_FEATURES)

    prompt = _PROMPT.format(text=text[:_TEXT_CHAR_LIMIT])
    try:
        response = await gemini_llm_func(prompt, system_prompt=_SYSTEM)
        raw = _parse_response(response)
        features = _validate(raw)
        return features
    except Exception as exc:
        print(f"[Pirelli] LLM抽出失敗 ({pdf_path.name}): {exc}")
        return dict(FALLBACK_FEATURES)


def extract(pdf_path: Path) -> dict[str, float | int]:
    """同期版ラッパー。通常スクリプト・Streamlitから呼び出す用。"""
    return asyncio.run(extract_async(pdf_path))
