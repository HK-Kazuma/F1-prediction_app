"""
FIA公式サイトからF1関連PDFを自動取得するモジュール。

対象文書:
  - Pirelli Preview          (毎レース: data/race_docs/{year}/round_{nn}_{name}/)
  - Race Director Event Notes (毎レース: 同上)
  - Technical Regulations    (年1回  : data/regulations/{year}/)

FIAサイトのURLパターン:
  https://www.fia.com/documents/championships/
    fia-formula-one-world-championship-14/season/season-{year}-{id}

JavaScriptレンダリングが必要な場合は自動取得に失敗する。
その場合は手動で上記ディレクトリにPDFを配置すればパイプラインは継続できる。
"""

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.constants import FIA_RACE_DOCS_DIR, FIA_REGULATIONS_DIR

_FIA_BASE = "https://www.fia.com"
_F1_DOCS_PATH = "/documents/championships/fia-formula-one-world-championship-14"

# FIAのシーズンページID（年ごとに変わる）
_SEASON_IDS: dict[int, int] = {
    2022: 2061,
    2023: 2066,
    2024: 2069,
    2025: 2071,
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# タイトルマッチング用キーワード
_PIRELLI_KEYWORDS = ["pirelli", "preview", "compound"]
_EVENT_NOTES_KEYWORDS = ["event notes", "race director", "director's notes"]
_TECH_REG_KEYWORDS = ["technical regulations", "technical reg"]

_DOWNLOAD_DELAY = 2.0  # FIAサーバーへの礼儀（秒）


# ---------- 内部ユーティリティ ----------

def _season_url(year: int) -> str:
    sid = _SEASON_IDS.get(year)
    if sid is None:
        raise ValueError(
            f"{year}年のシーズンIDが未登録です。"
            f"_SEASON_IDS に追加してください。既知の年: {sorted(_SEASON_IDS)}"
        )
    return f"{_FIA_BASE}{_F1_DOCS_PATH}/season/season-{year}-{sid}"


def _matches(title: str, keywords: list[str]) -> bool:
    t = title.lower()
    return any(kw in t for kw in keywords)


def _contains_round(title: str, round_number: int) -> bool:
    """タイトルにラウンド番号または「Round X」が含まれるか確認する。"""
    patterns = [
        str(round_number),
        f"round {round_number}",
        f"round {round_number:02d}",
    ]
    t = title.lower()
    return any(p in t for p in patterns)


def _fetch_doc_list(year: int) -> list[dict]:
    """FIAドキュメントページからPDFリンク一覧を取得する。"""
    url = _season_url(year)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[FIA] ページ取得失敗: {exc}")
        print(f"[FIA] 手動取得URL: {url}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    docs = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        # FIAのPDFリンクはファイル名が .pdf か /system/files/ を含む
        if not (".pdf" in href.lower() or "/system/files/" in href):
            continue
        title = (a.get_text(strip=True) or a.get("title", "")).strip()
        if not title:
            continue
        full_url = href if href.startswith("http") else _FIA_BASE + href
        docs.append({"title": title, "url": full_url})

    return docs


def _download(url: str, dest: Path) -> bool:
    """PDFをダウンロードして保存する。既存ファイルはスキップ。"""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        # Content-Type確認（HTMLが返ってきた場合はJSレンダリング必要）
        ct = resp.headers.get("Content-Type", "")
        if "html" in ct and "pdf" not in ct:
            print(f"[FIA] PDFではなくHTMLが返されました（JSレンダリング必要の可能性）: {url}")
            return False
        dest.write_bytes(resp.content)
        time.sleep(_DOWNLOAD_DELAY)
        return True
    except requests.RequestException as exc:
        print(f"[FIA] ダウンロード失敗 ({dest.name}): {exc}")
        return False


# ---------- 公開API ----------

def fetch_race_docs(
    year: int,
    round_number: int,
    event_name: str,
) -> dict[str, Path | None]:
    """
    指定ラウンドの Pirelli Preview と Race Director Notes を取得する。

    Args:
        year        : シーズン年
        round_number: ラウンド番号 (1始まり)
        event_name  : イベント名 (例: "Bahrain") — ディレクトリ名に使う

    Returns:
        {"pirelli": Path | None, "event_notes": Path | None}
        取得成功した場合はPathオブジェクト、失敗/未発見の場合はNone。
    """
    safe_name = re.sub(r"[^\w]", "_", event_name.lower())
    round_dir = (
        Path(FIA_RACE_DOCS_DIR)
        / str(year)
        / f"round_{round_number:02d}_{safe_name}"
    )
    round_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path | None] = {"pirelli": None, "event_notes": None}

    # 手動配置ファイルを先にチェック（自動取得より優先）
    manual_pirelli = round_dir / "pirelli_preview.pdf"
    manual_notes = round_dir / "event_notes.pdf"
    if manual_pirelli.exists():
        result["pirelli"] = manual_pirelli
    if manual_notes.exists():
        result["event_notes"] = manual_notes

    # 両方揃っていれば取得不要
    if result["pirelli"] and result["event_notes"]:
        return result

    docs = _fetch_doc_list(year)
    if not docs:
        _print_manual_instructions(round_dir, year, round_number)
        return result

    for doc in docs:
        title = doc["title"]
        if not _contains_round(title, round_number):
            continue
        if result["pirelli"] is None and _matches(title, _PIRELLI_KEYWORDS):
            dest = round_dir / "pirelli_preview.pdf"
            if _download(doc["url"], dest):
                result["pirelli"] = dest
                print(f"[FIA] Pirelli Preview 取得: {dest}")
        if result["event_notes"] is None and _matches(title, _EVENT_NOTES_KEYWORDS):
            dest = round_dir / "event_notes.pdf"
            if _download(doc["url"], dest):
                result["event_notes"] = dest
                print(f"[FIA] Event Notes 取得: {dest}")

    if result["pirelli"] is None or result["event_notes"] is None:
        _print_manual_instructions(round_dir, year, round_number)

    return result


def fetch_regulation_pdf(year: int) -> Path | None:
    """
    指定年の FIA Technical Regulations PDF を取得する。
    既にローカルにあればそのまま返す。

    Returns:
        PDF の Path、取得失敗時は None。
    """
    reg_dir = Path(FIA_REGULATIONS_DIR) / str(year)
    dest = reg_dir / "technical_regulations.pdf"

    if dest.exists():
        return dest

    docs = _fetch_doc_list(year)
    if not docs:
        print(f"[FIA] {year}年 Technical Regulations を手動で配置してください: {dest}")
        return None

    for doc in docs:
        if _matches(doc["title"], _TECH_REG_KEYWORDS):
            if _download(doc["url"], dest):
                print(f"[FIA] Technical Regulations 取得: {dest}")
                return dest

    print(f"[FIA] {year}年 Technical Regulations が見つかりませんでした。")
    print(f"[FIA] 手動で配置: {dest}")
    return None


def _print_manual_instructions(round_dir: Path, year: int, round_number: int) -> None:
    print(
        f"\n[FIA] 自動取得できなかった文書があります。"
        f"\n      以下のディレクトリに手動でPDFを配置してください:"
        f"\n      {round_dir}"
        f"\n      ファイル名:"
        f"\n        pirelli_preview.pdf  — Pirelli {year} Round {round_number} Preview"
        f"\n        event_notes.pdf      — Race Director Event Notes"
        f"\n      取得元: {_season_url(year)}\n"
    )
