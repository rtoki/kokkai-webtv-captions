"""議員名簿パーサ (members.py) のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kokkai.shugiin.members import (
    _discover_kaiha_list,
    _normalize_name,
    parse_kaiha_html,
)


FIXTURE = Path(__file__).parent / "fixtures" / "shugiin_kaiha_011.html"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"fixture 未取得: {FIXTURE.name} — `python scripts/fetch_fixtures.py` で取得",
)


@pytest.fixture(scope="module")
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# ---------- 会派ナビ発見 ----------

def test_discover_kaiha_list(html):
    nav = _discover_kaiha_list(html)
    codes = [c for c, _ in nav]
    assert "011" in codes
    assert "020" in codes
    assert "999" in codes  # 無所属
    assert len(nav) >= 4


def test_discover_kaiha_names_match_codes(html):
    nav = dict(_discover_kaiha_list(html))
    assert "自由民主党" in nav["011"]
    assert "無所属" in nav["999"]


# ---------- 名前正規化 ----------

@pytest.mark.parametrize("raw,clean_expected", [
    ("逢沢　　一郎君", "逢沢一郎"),
    ("逢沢　　一郎君\n", "逢沢一郎"),
    ("あかま　二郎君", "あかま二郎"),
    ("青山　　繁晴君\n", "青山繁晴"),
    ("山田太郎", "山田太郎"),  # 既に「君」無し
])
def test_normalize_name(raw, clean_expected):
    clean, _with_spaces = _normalize_name(raw)
    assert clean == clean_expected


def test_normalize_name_preserves_spaced_form():
    clean, with_spaces = _normalize_name("逢沢　　一郎君")
    assert clean == "逢沢一郎"
    assert "　" in with_spaces  # 全角スペース保持
    assert "君" not in with_spaces


# ---------- 会派 HTML パース ----------

def test_parse_kaiha_html_member_count(html):
    members = parse_kaiha_html(html, "自由民主党・無所属の会")
    # 011 = 自由民主党、~300 名規模 (実値 316)
    assert 200 < len(members) < 400


def test_parse_kaiha_html_first_member(html):
    members = parse_kaiha_html(html, "自由民主党・無所属の会")
    # 五十音順なので最初は「逢沢」「青山」あたり
    first = members[0]
    assert first["name"] == "逢沢一郎"
    assert first["faction"] == "自由民主党・無所属の会"
    assert first["furigana"] == "あいさわ いちろう"
    assert first["district"] == "岡山1"


def test_parse_kaiha_html_skips_header_row(html):
    members = parse_kaiha_html(html, "自由民主党・無所属の会")
    # 「氏名」「ふりがな」「選挙区」のヘッダ行が混ざっていないことを確認
    assert all(m["name"] != "氏名" for m in members)
    assert all(m["furigana"] != "ふりがな" for m in members)
    assert all(m["district"] != "選挙区" for m in members)


def test_parse_kaiha_html_all_have_required_fields(html):
    members = parse_kaiha_html(html, "自由民主党・無所属の会")
    for m in members:
        assert m["name"]
        assert m["faction"] == "自由民主党・無所属の会"
        # furigana と district は基本的にあるはず (まれに空欄も許容)
        assert "furigana" in m
        assert "district" in m


def test_parse_kaiha_html_name_no_kun_suffix(html):
    members = parse_kaiha_html(html, "自由民主党・無所属の会")
    assert all(not m["name"].endswith("君") for m in members)
