"""initial_prompt builder (hints.py) のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kokkai.shugiin.extract import parse_meta_html
from kokkai.shugiin.hints import (
    _extract_proper_nouns_from_agenda,
    build_initial_prompt,
)


FIXTURE = Path(__file__).parent / "fixtures" / "shugiin_deli56246.html"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"fixture 未取得: {FIXTURE.name} — `python scripts/fetch_fixtures.py` で取得",
)


@pytest.fixture(scope="module")
def meta() -> dict:
    html = FIXTURE.read_text(encoding="utf-8")
    _, m = parse_meta_html(html, "56246")
    return m


# ---------- 議題からの固有名詞抽出 ----------

def test_extract_proper_nouns_from_agenda_basic():
    nouns = _extract_proper_nouns_from_agenda([
        "経済施策を一体的に講ずることによる安全保障の確保の推進に関する法律案"
    ])
    assert "経済施策" in nouns
    assert "安全保障" in nouns


def test_extract_proper_nouns_excludes_functional_phrases():
    nouns = _extract_proper_nouns_from_agenda(["AAAに関するBBBの法律"])
    assert "に関する" not in nouns


def test_extract_proper_nouns_dedups():
    nouns = _extract_proper_nouns_from_agenda([
        "経済施策の法律", "経済施策の改正"
    ])
    # 「経済施策」は重複しても 1 回だけ
    assert nouns.count("経済施策") == 1


# ---------- build_initial_prompt 全体 ----------

def test_build_initial_prompt_includes_meeting(meta):
    p = build_initial_prompt(meta, members=None, max_chars=500)
    assert "内閣委員会" in p
    assert "2026-05-15" in p


def test_build_initial_prompt_includes_all_unique_speakers(meta):
    p = build_initial_prompt(meta, members=None, max_chars=500)
    speaker_names = {s["name"] for s in meta["speakers"]}
    # 全ユニーク発言者名がプロンプトに含まれる
    for n in speaker_names:
        assert n in p, f"speaker {n} not in prompt"


def test_build_initial_prompt_dedups_repeated_speakers(meta):
    p = build_initial_prompt(meta, members=None, max_chars=500)
    # 発言者リストに同一人物が複数回出現するケース (途中で委員長が割り込む等) でも、
    # プロンプトには 1 回しか入らない (dedup される)
    speakers = meta["speakers"]
    name_counts: dict[str, int] = {}
    for s in speakers:
        name_counts[s["name"]] = name_counts.get(s["name"], 0) + 1
    repeated = [n for n, c in name_counts.items() if c > 1]
    assert repeated, "fixture には同一人物が複数回出現する発言者がいるはず"
    speakers_line = next(
        line for line in p.splitlines() if line.startswith("発言者:")
    )
    for n in repeated:
        assert speakers_line.count(n) == 1, \
            f"{n!r} がプロンプト内で dedup されていない"


def test_build_initial_prompt_respects_max_chars(meta):
    p = build_initial_prompt(meta, members=None, max_chars=80)
    assert len(p) <= 80


def test_build_initial_prompt_drops_low_priority_first(meta):
    p_short = build_initial_prompt(meta, members=None, max_chars=80)
    p_long = build_initial_prompt(meta, members=None, max_chars=500)
    # 短い方は議題や用語が落ちて、会議名・発言者は残る
    assert "内閣委員会" in p_short
    # 長い方には Tier 4 (議題) や Tier 6 (用語) が入る
    assert ("議題" in p_long) or ("用語" in p_long)


def test_build_initial_prompt_includes_factions(meta):
    p = build_initial_prompt(meta, members=None, max_chars=500)
    assert "会派:" in p
    # 「内閣委員長」みたいな役職は会派欄から除外される
    factions_line = next(
        line for line in p.splitlines() if line.startswith("会派:")
    )
    assert "内閣委員長" not in factions_line


def test_build_initial_prompt_with_members(meta):
    # ダミー member 数件で関連議員セクションが付くことを確認
    dummy_members = [
        {"name": "テスト太郎", "faction": "自由民主党・無所属の会"},
        {"name": "テスト次郎", "faction": "日本維新の会"},
        {"name": "発言者本人", "faction": "自由民主党・無所属の会"},
    ]
    p = build_initial_prompt(meta, members=dummy_members, max_chars=500)
    # 関連議員セクションが追加されている
    assert "関連議員" in p or "テスト" in p


def test_build_initial_prompt_no_speakers():
    minimal_meta = {
        "title": "予算委員会",
        "date": "2026-01-01",
        "speakers": [],
        "agenda": [],
        "deli_id": "1",
        "page_url": "x",
    }
    p = build_initial_prompt(minimal_meta, members=None, max_chars=200)
    assert "予算委員会" in p
    # 発言者セクションは無し
    assert "発言者:" not in p
