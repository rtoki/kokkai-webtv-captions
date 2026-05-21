"""kokkai/search の純関数テスト (ネット I/O なし、SudachiPy + 自前 BM25 のみ)。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kokkai.search.index import _parse_base_filename, _parse_vtt, iter_records
from kokkai.search.query import filter_records, score_records
from kokkai.search.render import _days_since, _format_time, _jump_url
from kokkai.search.tokenize import tokenize, tokenize_query


# ---------- tokenize ----------

def test_tokenize_kanji_compound():
    """漢字熟語が短単位に分解されること。"""
    toks = tokenize("成年後見の制度")
    assert "成年" in toks
    assert "後見" in toks
    assert "制度" in toks
    assert "の" not in toks  # 助詞は除外


def test_tokenize_katakana_word():
    """カタカナ語 (Sudachi が「形状詞」「名詞」等に分類するもの) が拾われる。"""
    toks = tokenize("サイバーセキュリティ対策")
    text = "".join(toks)
    assert "サイバー" in text or "サイバ" in text


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_query_phrase():
    terms, phrases = tokenize_query('成年後見 "事理弁識" 制度')
    assert "事理弁識" in phrases
    assert any(t in terms for t in ["成年", "後見", "制度"])


def test_tokenize_query_no_phrase():
    terms, phrases = tokenize_query("成年後見 制度")
    assert phrases == []
    assert "後見" in terms


# ---------- index._parse_base_filename ----------

@pytest.mark.parametrize("base,expected_date,expected_title", [
    # 新形式 (衆/参 漢字 1 字 + ID)
    ("2026-05-15_法務委員会_衆56245", "2026-05-15", "法務委員会"),
    ("2026年4月15日_デジタル委員会_参8955", "2026年4月15日", "デジタル委員会"),
    # 後方互換: 旧 _shugiin / _sangiin / _deli 形式
    ("2026-05-15_法務委員会_shugiin56245", "2026-05-15", "法務委員会"),
    ("2026年4月15日_デジタル委員会_sangiin8955", "2026年4月15日", "デジタル委員会"),
    ("2026-05-14_本会議_deli56237", "2026-05-14", "本会議"),
    # ID なし (旧 sangiin)
    ("2026年4月15日_内閣委員会", "2026年4月15日", "内閣委員会"),
    ("2025年5月9日_災害対策特別委員会", "2025年5月9日", "災害対策特別委員会"),
])
def test_parse_base_filename(base, expected_date, expected_title):
    date, title = _parse_base_filename(base)
    assert date == expected_date
    assert title == expected_title


def test_parse_base_filename_no_match():
    date, title = _parse_base_filename("random_filename_no_date")
    assert date == ""


# ---------- index._parse_vtt ----------

def test_parse_vtt_basic():
    vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\n"
        "一つ目の発言です。\n\n"
        "00:00:05.000 --> 00:00:08.000\n"
        "二つ目の発言です。\n"
    )
    cues = _parse_vtt(vtt)
    assert len(cues) == 2
    assert cues[0]["text"] == "一つ目の発言です。"
    assert cues[0]["start"] == 1.0
    assert cues[1]["start"] == 5.0


def test_parse_vtt_skips_header():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nテスト\n"
    cues = _parse_vtt(vtt)
    assert len(cues) == 1
    assert cues[0]["text"] == "テスト"


# ---------- query.score_records ----------

def _make_record(text: str, **kwargs) -> dict:
    return {
        "house": "sangiin", "id": "1", "date": "2026-05-15",
        "title": "テスト委員会", "page_url": "https://example.com/?sid=1",
        "cue_idx": 0, "start": 0.0, "end": 1.0, "text": text,
        "speaker_name": "テスト議員", "speaker_group": "テスト党",
        "meta_path": "/tmp/x.meta.json", **kwargs,
    }


def test_score_records_basic():
    records = [
        _make_record("成年後見の制度について議論しました。"),
        _make_record("税制改革の話。"),
        _make_record("関係ない話。"),
    ]
    terms, phrases = tokenize_query("成年後見")
    scored = score_records(records, terms, phrases)
    assert len(scored) >= 1
    assert "成年後見" in scored[0][0]["text"]


def test_score_records_phrase_filter():
    """phrase が含まれないレコードは除外される。"""
    records = [
        _make_record("成年後見の制度を議論しました。"),
        _make_record("成年は重要ですが後見は別の話です。"),
    ]
    terms, phrases = tokenize_query('"成年後見"')
    scored = score_records(records, terms, phrases)
    # phrase "成年後見" が連結文字列で含まれる方のみヒット
    assert all("成年後見" in r["text"] for r, _ in scored)


def test_score_records_empty_query():
    assert score_records([_make_record("テスト")], [], []) == []


def test_score_records_empty_records():
    assert score_records([], ["成年"], []) == []


# ---------- query.filter_records ----------

def test_filter_records_by_date():
    records = [
        _make_record("a", date="2026-05-01"),
        _make_record("b", date="2026-05-15"),
        _make_record("c", date="2026-06-01"),
    ]
    out = filter_records(records, since="2026-05-10", until="2026-05-20")
    assert len(out) == 1
    assert out[0]["text"] == "b"


def test_filter_records_by_date_japanese_format():
    """「2026年5月15日」形式も範囲指定で扱える。"""
    records = [
        _make_record("a", date="2026年4月15日"),
        _make_record("b", date="2026年5月15日"),
    ]
    out = filter_records(records, since="2026-05-01")
    assert len(out) == 1
    assert out[0]["text"] == "b"


def test_filter_records_by_speaker():
    records = [
        _make_record("a", speaker_name="山田太郎"),
        _make_record("b", speaker_name="鈴木花子"),
        _make_record("c", speaker_name=None),
    ]
    out = filter_records(records, speaker="山田")
    assert len(out) == 1
    assert out[0]["text"] == "a"


def test_filter_records_by_committee():
    records = [
        _make_record("a", title="法務委員会"),
        _make_record("b", title="内閣委員会"),
    ]
    out = filter_records(records, committee="法務")
    assert len(out) == 1
    assert out[0]["text"] == "a"


def test_filter_records_by_house():
    records = [
        _make_record("a", house="sangiin"),
        _make_record("b", house="shugiin"),
    ]
    out = filter_records(records, house="shugiin")
    assert len(out) == 1
    assert out[0]["text"] == "b"


# ---------- render utilities ----------

def test_format_time():
    assert _format_time(0) == "00:00:00"
    assert _format_time(61) == "00:01:01"
    assert _format_time(3661) == "01:01:01"


def test_jump_url_sangiin():
    r = _make_record("x", house="sangiin", page_url="https://example.com/?sid=1", start=100.5)
    url = _jump_url(r)
    assert url.endswith("#100")


def test_jump_url_shugiin():
    r = _make_record("x", house="shugiin", page_url="https://example.com/?deli_id=2", start=200.7)
    url = _jump_url(r)
    assert "time=200" in url


def test_days_since():
    # 過去の日付なら正の値
    assert _days_since("2020-01-01") is not None
    assert _days_since("2020-01-01") > 0
    # 不正フォーマット
    assert _days_since("不明") is None
    assert _days_since("") is None


# ---------- iter_records 統合 (tmp_path) ----------

def test_iter_records_meta_path(tmp_path: Path):
    """meta.json + vtt 経由の正常系。"""
    base = "2026-05-15_テスト委員会"
    (tmp_path / f"{base}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト発言です。\n",
        encoding="utf-8",
    )
    (tmp_path / f"{base}.meta.json").write_text(
        json.dumps({
            "house": "sangiin",
            "id": "9999",
            "date": "2026-05-15",
            "title": "テスト委員会",
            "page_url": "https://example.com/?sid=9999",
            "speakers": [{"start": 0.0, "name": "山田", "group": "テスト党"}],
            "files": {"vtt": f"{base}.vtt"},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    records = iter_records(tmp_path)
    assert len(records) == 1
    r = records[0]
    assert r["house"] == "sangiin"
    assert r["title"] == "テスト委員会"
    assert r["speaker_name"] == "山田"
    assert "テスト発言" in r["text"]


def test_iter_records_vtt_fallback(tmp_path: Path):
    """meta.json が無くても vtt 単独で取り込まれる (legacy fallback)。"""
    base = "2026-05-15_法務委員会"
    (tmp_path / f"{base}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテキスト\n",
        encoding="utf-8",
    )
    records = iter_records(tmp_path)
    assert len(records) == 1
    assert records[0]["title"] == "法務委員会"
    assert records[0]["date"] == "2026-05-15"
    # speaker info は fallback では取れない
    assert records[0]["speaker_name"] is None


def test_iter_records_empty_dir(tmp_path: Path):
    assert iter_records(tmp_path) == []


# ---------- cache ----------

def test_cache_creates_file_on_first_run(tmp_path: Path):
    from kokkai.search import cache as _cache
    base = "2026-05-15_テスト委員会"
    (tmp_path / f"{base}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト発言\n",
        encoding="utf-8",
    )
    assert not _cache.cache_path(tmp_path).exists()
    records = iter_records(tmp_path)
    assert len(records) == 1
    # キャッシュファイルが作られる
    assert _cache.cache_path(tmp_path).exists()


def test_cache_hit_skips_retokenize(tmp_path: Path, monkeypatch):
    """キャッシュが新鮮なら tokenize() は呼ばれない。"""
    base = "2026-05-15_テスト委員会"
    src = tmp_path / f"{base}.vtt"
    src.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n初回タイムテキスト\n",
        encoding="utf-8",
    )

    # 1 回目: キャッシュ作成
    iter_records(tmp_path)

    # 2 回目: tokenize を mock してそれが呼ばれないことを確認
    from kokkai.search import index as idx
    call_count = {"n": 0}
    orig_tokenize = idx.tokenize

    def counting_tokenize(text):
        call_count["n"] += 1
        return orig_tokenize(text)

    monkeypatch.setattr(idx, "tokenize", counting_tokenize)
    iter_records(tmp_path)
    assert call_count["n"] == 0, "キャッシュヒット時に tokenize が呼ばれてはならない"


def test_cache_invalidated_on_mtime_change(tmp_path: Path):
    """ソースファイルが書き換えられたらキャッシュは無効化される。"""
    import time
    from kokkai.search import cache as _cache

    base = "2026-05-15_テスト委員会"
    src = tmp_path / f"{base}.vtt"
    src.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n最初の発言\n",
        encoding="utf-8",
    )
    iter_records(tmp_path)

    # ファイル書き換え (mtime 変化)
    time.sleep(0.01)
    src.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n変更後の発言\n",
        encoding="utf-8",
    )

    records = iter_records(tmp_path)
    assert len(records) == 1
    assert "変更後" in records[0]["text"]


def test_cache_disabled(tmp_path: Path):
    """use_cache=False ならキャッシュファイルを作らない。"""
    from kokkai.search import cache as _cache
    base = "2026-05-15_テスト委員会"
    (tmp_path / f"{base}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト\n",
        encoding="utf-8",
    )
    iter_records(tmp_path, use_cache=False)
    assert not _cache.cache_path(tmp_path).exists()


def test_cache_clear(tmp_path: Path):
    from kokkai.search import cache as _cache
    base = "2026-05-15_テスト委員会"
    (tmp_path / f"{base}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト\n",
        encoding="utf-8",
    )
    iter_records(tmp_path)
    assert _cache.cache_path(tmp_path).exists()
    assert _cache.clear_cache(tmp_path) is True
    assert not _cache.cache_path(tmp_path).exists()
    assert _cache.clear_cache(tmp_path) is False  # 2 回目は False


def test_cache_version_mismatch_rebuilds(tmp_path: Path):
    """version 不一致のキャッシュは無視されて再構築される。"""
    import json as _json
    from kokkai.search import cache as _cache

    base = "2026-05-15_テスト委員会"
    (tmp_path / f"{base}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト\n",
        encoding="utf-8",
    )
    # 古い version のキャッシュを置く
    _cache.cache_path(tmp_path).write_text(
        _json.dumps({"version": 999, "files": {"bogus": {"mtime": 0, "cues": []}}}),
        encoding="utf-8",
    )
    records = iter_records(tmp_path)
    assert len(records) == 1
    # 再構築された結果、新しい version になっている
    saved = _json.loads(_cache.cache_path(tmp_path).read_text(encoding="utf-8"))
    assert saved["version"] == _cache.CACHE_VERSION


def test_cache_picks_up_new_file_after_first_run(tmp_path: Path):
    """初回構築後に新ファイルが追加されたら、2 回目の iter_records で manifest drift を
    検知してキャッシュに反映する (今回 9027 を取り損ねた症状の回帰防止)。"""
    import json as _json
    from kokkai.search import cache as _cache

    # 1 回目: 1 ファイルだけある状態でキャッシュ構築
    (tmp_path / "2026-05-19_A委員会.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト A\n",
        encoding="utf-8",
    )
    r1 = iter_records(tmp_path)
    assert len(r1) == 1
    saved1 = _json.loads(_cache.cache_path(tmp_path).read_text(encoding="utf-8"))
    assert len(saved1.get("manifest") or []) == 1

    # 2 回目: 新規ファイル追加 → キャッシュは既存 mtime 比較で不変だが、
    # manifest drift で新ファイルが取り込まれる必要がある
    (tmp_path / "2026-05-21_B委員会.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト B\n",
        encoding="utf-8",
    )
    r2 = iter_records(tmp_path)
    assert len(r2) == 2
    saved2 = _json.loads(_cache.cache_path(tmp_path).read_text(encoding="utf-8"))
    assert len(saved2.get("manifest") or []) == 2
    assert any("B委員会" in p for p in saved2["manifest"])


def test_cache_drops_dead_entries_on_file_removal(tmp_path: Path):
    """ソースファイルが消えたら次回 iter_records でキャッシュから drop される。"""
    import json as _json
    from kokkai.search import cache as _cache

    (tmp_path / "2026-05-19_A委員会.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト A\n",
        encoding="utf-8",
    )
    (tmp_path / "2026-05-20_B委員会.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nテスト B\n",
        encoding="utf-8",
    )
    iter_records(tmp_path)
    saved1 = _json.loads(_cache.cache_path(tmp_path).read_text(encoding="utf-8"))
    assert len(saved1["files"]) == 2

    # B を消す
    (tmp_path / "2026-05-20_B委員会.vtt").unlink()
    iter_records(tmp_path)
    saved2 = _json.loads(_cache.cache_path(tmp_path).read_text(encoding="utf-8"))
    assert len(saved2["files"]) == 1
    assert all("B委員会" not in p for p in saved2["files"].keys())
