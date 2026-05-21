"""衆議院 extract.py の純関数 (ネットなし、フィクスチャベース) のテスト."""

from __future__ import annotations

from pathlib import Path

import pytest

from kokkai.shugiin.extract import (
    _extract_agenda,
    _extract_date,
    _extract_hls_url,
    _extract_meeting_name,
    _extract_speakers,
    _parse_deli_id,
    parse_meta_html,
)


FIXTURE = Path(__file__).parent / "fixtures" / "shugiin_deli56246.html"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"fixture 未取得: {FIXTURE.name} — `python scripts/fetch_fixtures.py` で取得",
)


@pytest.fixture(scope="module")
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# ---------- deli_id 抽出 ----------

@pytest.mark.parametrize("target,expected", [
    ("56246", "56246"),
    ("  56246  ", "56246"),
    ("https://www.shugiintv.go.jp/jp/index.php?ex=VL&deli_id=56246&media_type=", "56246"),
    ("index.php?ex=VL&deli_id=12345", "12345"),
])
def test_parse_deli_id(target, expected):
    assert _parse_deli_id(target) == expected


def test_parse_deli_id_invalid():
    with pytest.raises(ValueError):
        _parse_deli_id("abc")


# ---------- HLS URL 抽出 ----------

def test_extract_hls_url(html):
    url = _extract_hls_url(html)
    assert url is not None
    assert url.startswith("https://")
    assert url.endswith(".m3u8")
    assert "shugiintv.go.jp" in url


def test_extract_hls_url_http_upgraded_to_https(html):
    url = _extract_hls_url(html)
    assert url.startswith("https://")
    # 元の HTML は http:// を含む
    assert 'value="http://hlsvod.shugiintv.go.jp' in html


def test_extract_hls_url_missing():
    assert _extract_hls_url("<html>no player</html>") is None


# ---------- 日付・会議名 ----------

def test_extract_date(html):
    assert _extract_date(html) == "2026-05-15"


def test_extract_date_pad_single_digit():
    assert _extract_date("2024年1月3日 開会") == "2024-01-03"


def test_extract_date_missing():
    assert _extract_date("no date here") == ""


def test_extract_meeting_name(html):
    assert _extract_meeting_name(html) == "内閣委員会"


@pytest.mark.parametrize("snippet,expected", [
    ('<td>本会議 (3時間)</td>', "本会議"),
    ('<td>予算委員会 (5時間30分)</td>', "予算委員会"),
    ('<td>憲法審査会 (1時間)</td>', "憲法審査会"),
    ('<td>調査会 (30分)</td>', "調査会"),
    ('<td>公聴会 (2時間)</td>', "公聴会"),
])
def test_extract_meeting_name_variants(snippet, expected):
    assert _extract_meeting_name(snippet) == expected


# ---------- 発言者一覧 ----------

def test_extract_speakers_count(html):
    speakers = _extract_speakers(html)
    # フィクスチャは 13 件の time=N.N リンクを持つ (内部委員長 2 回登場含む、dedup後でも 13 個ある)
    assert len(speakers) >= 10


def test_extract_speakers_sorted_by_start(html):
    speakers = _extract_speakers(html)
    starts = [s["start"] for s in speakers]
    assert starts == sorted(starts)


def test_extract_speakers_first_is_chair(html):
    speakers = _extract_speakers(html)
    # 委員会の冒頭は委員長 (実 fixture の構造的特性をチェック)
    assert "委員長" in speakers[0]["group"]
    assert speakers[0]["name"]


def test_extract_speakers_parses_name_and_group(html):
    speakers = _extract_speakers(html)
    # 「(<会派名>)」の括弧パース成否: name と group の両方が空でないものが存在
    parsed = [s for s in speakers if s["name"] and s["group"]]
    assert len(parsed) >= 3
    # 政党/会派ラベルらしき文字列 (・含む) が group に出現していること
    assert any("・" in s["group"] or "党" in s["group"] for s in parsed)


def test_extract_speakers_excludes_play_from_start():
    snippet = (
        '<A HREF="/jp/index.php?ex=VL&deli_id=99999&time=10.0">はじめから再生</A>'
        '<A HREF="/jp/index.php?ex=VL&deli_id=99999&time=20.0">山田太郎(委員長)</a>'
    )
    speakers = _extract_speakers(snippet)
    names = [s["name"] for s in speakers]
    assert "はじめから再生" not in names
    assert "山田太郎" in names


# ---------- parse_meta_html (組み合わせ) ----------

def test_parse_meta_html_full(html):
    m3u8, meta = parse_meta_html(html, "56246")
    assert m3u8 is not None
    assert meta["deli_id"] == "56246"
    assert meta["date"] == "2026-05-15"
    assert meta["title"] == "内閣委員会"
    assert len(meta["speakers"]) >= 10
    assert meta["page_url"].endswith("deli_id=56246&media_type=")
    assert isinstance(meta["agenda"], list)


# ---------- 議題 (法案名) 抽出 ----------

def test_extract_agenda(html):
    items = _extract_agenda(html)
    assert len(items) >= 1
    # フィクスチャの議題: 経済施策を一体的に...の法律案
    assert any("経済施策" in x for x in items)
    assert any("法律案" in x for x in items)


def test_extract_agenda_strips_image_tags(html):
    items = _extract_agenda(html)
    for it in items:
        assert "<" not in it
        assert "spacer" not in it.lower()


def test_extract_agenda_empty():
    assert _extract_agenda("<html>no agenda</html>") == []
