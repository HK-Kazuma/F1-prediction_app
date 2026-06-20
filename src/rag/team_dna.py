"""
チームDNAスコア生成モジュール（Step 3）。

LightRAGのKGに蓄積した規制文書の知識を使い、
「この年の規制変更はどのチームに有利か」を 0.0〜1.0 で定量化する。

スコアの意味:
  1.0 = 規制変更がチームの技術的強みと完全に一致（非常に有利）
  0.5 = 中立（影響なし）
  0.0 = 規制変更がチームに不利

生成フロー:
  1. KGに「{year}年の規制の主な変更点は？」とクエリ
  2. 返ってきたコンテキスト + チーム情報をGeminiに渡す
  3. チームごとのスコアをJSONで取得
  4. JSON を {TEAM_DNA_SCORES_PATH}_{year}.json にキャッシュ

LLM呼び出しはコストがかかる（Gemini無料枠では時間もかかる）ため、
一度生成したスコアはキャッシュから読み込む。
"""

import asyncio
import json
import re
from pathlib import Path

from src.constants import GEMINI_RATE_LIMIT_SLEEP_SECONDS, TEAM_DNA_SCORES_PATH
from src.rag.ingest import query_async
from src.rag.llm import gemini_llm_func

# F1に参戦している主要コンストラクター名（FastF1 のConstructor列と一致させる）
KNOWN_CONSTRUCTORS = [
    "Red Bull",
    "Mercedes",
    "Ferrari",
    "McLaren",
    "Aston Martin",
    "Alpine",
    "Williams",
    "RB",
    "Haas",
    "Kick Sauber",
]

_TEAM_DNA_PROMPT = """\
You are an F1 technical analyst. Based on the FIA regulation context below,
rate how favorable the {year} regulation changes are for {team}.

Scoring criteria:
- Consider the team's historical engineering strengths and philosophy
- Consider how similar regulation changes benefited or hurt this team in past seasons
- Focus on the specific technical areas emphasized in {year}

Context from FIA {year} Technical Regulations:
{context}

Respond ONLY with JSON:
{{"dna_score": <float 0.0-1.0>, "reasoning": "<one concise sentence>"}}

Where: 1.0=strongly favors {team}, 0.5=neutral, 0.0=strongly disfavors {team}
"""

_REG_QUESTION = (
    "What are the most significant technical regulation changes in {year}? "
    "Which engineering areas (aerodynamics, power unit, weight, suspension) "
    "are most affected and how?"
)


def _parse_json(response: str) -> dict:
    match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
    if not match:
        raise ValueError(f"JSONが見つかりません: {response[:200]}")
    return json.loads(match.group())


async def _score_one_team(team: str, year: int, reg_context: str) -> float:
    prompt = _TEAM_DNA_PROMPT.format(
        year=year,
        team=team,
        context=reg_context[:2000],
    )
    try:
        response = await gemini_llm_func(prompt, system_prompt="Respond with JSON only.")
        data = _parse_json(response)
        score = float(max(0.0, min(1.0, data.get("dna_score", 0.5))))
        reasoning = data.get("reasoning", "")
        print(f"  {team:20s}: {score:.2f}  — {reasoning}")
        return score
    except Exception as exc:
        print(f"  {team:20s}: スコア生成失敗 ({exc}) → 0.5（中立）を使用")
        return 0.5


async def generate_async(
    year: int,
    constructors: list[str] | None = None,
) -> dict[str, float]:
    """
    指定年の全チームDNAスコアを生成して返す（非同期版）。

    Returns:
        {"Red Bull": 0.82, "Mercedes": 0.61, ...}
    """
    teams = constructors or KNOWN_CONSTRUCTORS
    print(f"\n[TeamDNA] {year}年のチームDNAスコアを生成中（{len(teams)}チーム）...")

    # まずKGから規制の文脈情報を1回取得する（全チーム共通）
    question = _REG_QUESTION.format(year=year)
    reg_context = await query_async(question, mode="global")
    print(f"[TeamDNA] 規制コンテキスト取得完了 ({len(reg_context)}文字)")

    scores: dict[str, float] = {}
    for team in teams:
        scores[team] = await _score_one_team(team, year, reg_context)
        # Gemini無料枠 15 RPM のレート制限対策
        await asyncio.sleep(GEMINI_RATE_LIMIT_SLEEP_SECONDS)

    return scores


def generate(year: int, constructors: list[str] | None = None) -> dict[str, float]:
    """同期版ラッパー。"""
    return asyncio.run(generate_async(year, constructors))


def load_or_generate(year: int) -> dict[str, float]:
    """
    キャッシュJSONがあれば読み込み、なければ生成してキャッシュに保存する。

    キャッシュファイル: data/processed/team_dna_scores_{year}.json
    """
    cache_path = _cache_path(year)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            scores = json.load(f)
        print(f"[TeamDNA] キャッシュから読み込み: {cache_path}")
        return scores

    scores = generate(year)
    _save_cache(scores, year)
    return scores


def _cache_path(year: int) -> Path:
    base = TEAM_DNA_SCORES_PATH.replace(".json", f"_{year}.json")
    return Path(base)


def _save_cache(scores: dict[str, float], year: int) -> None:
    path = _cache_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"[TeamDNA] スコアを保存: {path}")
