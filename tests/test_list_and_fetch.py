"""kokkai.list と kokkai.fetch の純関数テスト (ネット I/O なし)。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from kokkai import fetch as _fetch_mod
from kokkai.fetch import _classify, _parse_date
from kokkai.list.render import _weekday, render_human, render_json, render_jsonl
from kokkai.list import sangiin_sid
from kokkai.list.status import annotate, load_fetched_index


# ---------- fetch._classify (院判定) ----------

@pytest.mark.parametrize("target,expected", [
    ("8955", "sangiin"),         # 4 桁 sid
    ("9012", "sangiin"),
    ("19999", "sangiin"),        # 5 桁の sid 上限想定
    ("56239", "shugiin"),        # 5 桁 deli_id
    ("56246", "shugiin"),
    ("30000", "shugiin"),        # 境界
    ("https://www.webtv.sangiin.go.jp/webtv/detail.php?sid=8955", "sangiin"),
    ("https://www.shugiintv.go.jp/jp/index.php?ex=VL&deli_id=56246", "shugiin"),
])
def test_classify(target, expected):
    assert _classify(target) == expected


def test_classify_unknown():
    with pytest.raises(ValueError):
        _classify("not-a-number")


# ---------- fetch._parse_date / --from/--to ----------

def test_parse_date_formats():
    assert _parse_date("2026-04-20") == date(2026, 4, 20)
    assert _parse_date("2026/04/20") == date(2026, 4, 20)
    assert _parse_date("20260420") == date(2026, 4, 20)
    assert _parse_date("2026年4月20日") == date(2026, 4, 20)


def test_parse_date_invalid():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_date("not-a-date")


def test_fetch_requires_target_or_date():
    with pytest.raises(SystemExit):
        _fetch_mod.main([])


def _fake_items(*ids: tuple[str, str, str]) -> list[dict]:
    """``(house, id, date)`` 群を fetch_for_range 風 dict にする。"""
    return [
        {"house": h, "id": rid, "date": d, "title": f"会議{rid}",
         "page_url": f"https://example.com/{rid}"}
        for h, rid, d in ids
    ]


def test_enumerate_by_date_calls_both_houses(monkeypatch):
    """``--house`` 省略時は両院列挙、id 解決済のみ抽出。"""
    import argparse
    from kokkai.list import shugiin_list, sangiin_list

    monkeypatch.setattr(
        shugiin_list, "fetch_for_range",
        lambda s, e: _fake_items(("shugiin", "56239", "2026-04-21")),
    )
    monkeypatch.setattr(
        sangiin_list, "fetch_for_range",
        lambda s, e: _fake_items(
            ("sangiin", "8955", "2026-04-22"),
            ("sangiin", "", "2026-04-22"),  # id 未解決 → 除外される
        ),
    )
    args = argparse.Namespace(
        date_from=date(2026, 4, 20), date_to=date(2026, 4, 25), house=None,
    )
    out = _fetch_mod._enumerate_by_date(args)
    assert sorted(rid for _h, rid, _l in out) == ["56239", "8955"]


def test_enumerate_by_date_house_filter(monkeypatch):
    import argparse
    from kokkai.list import shugiin_list, sangiin_list

    called = {"sangiin": False, "shugiin": False}

    def fake_shugiin(s, e):
        called["shugiin"] = True
        return _fake_items(("shugiin", "56239", "2026-04-21"))

    def fake_sangiin(s, e):
        called["sangiin"] = True
        return _fake_items(("sangiin", "8955", "2026-04-22"))

    monkeypatch.setattr(shugiin_list, "fetch_for_range", fake_shugiin)
    monkeypatch.setattr(sangiin_list, "fetch_for_range", fake_sangiin)

    args = argparse.Namespace(
        date_from=date(2026, 4, 20), date_to=date(2026, 4, 25), house="shugiin",
    )
    out = _fetch_mod._enumerate_by_date(args)
    assert called == {"shugiin": True, "sangiin": False}
    assert [rid for _h, rid, _l in out] == ["56239"]


def test_dry_run_emits_enumerated_ids_only(monkeypatch, capsys):
    """``--dry-run`` 時は sangiin/shugiin の main を一切呼ばずに列挙のみ stdout に。"""
    from kokkai.list import shugiin_list, sangiin_list

    monkeypatch.setattr(
        shugiin_list, "fetch_for_range",
        lambda s, e: _fake_items(("shugiin", "56239", "2026-04-21")),
    )
    monkeypatch.setattr(
        sangiin_list, "fetch_for_range",
        lambda s, e: _fake_items(("sangiin", "8955", "2026-04-22")),
    )

    def explode(*a, **kw):
        raise AssertionError("fetch main should not be called in dry-run")

    monkeypatch.setattr("kokkai.sangiin.__main__.main", explode)
    monkeypatch.setattr("kokkai.shugiin.__main__.main", explode)

    with pytest.raises(SystemExit) as ei:
        _fetch_mod.main(["--from", "2026-04-20", "--to", "2026-04-25", "--dry-run"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "shugiin\t56239" in out
    assert "sangiin\t8955" in out


def test_dry_run_includes_explicit_targets(monkeypatch, capsys):
    """``--dry-run`` は --from/--to の列挙に加えて explicit target も stdout に出す。"""
    from kokkai.list import shugiin_list, sangiin_list

    monkeypatch.setattr(
        shugiin_list, "fetch_for_range",
        lambda s, e: _fake_items(("shugiin", "56239", "2026-04-21")),
    )
    monkeypatch.setattr(sangiin_list, "fetch_for_range", lambda s, e: [])

    def explode(*a, **kw):
        raise AssertionError("fetch main should not be called in dry-run")

    monkeypatch.setattr("kokkai.sangiin.__main__.main", explode)
    monkeypatch.setattr("kokkai.shugiin.__main__.main", explode)

    with pytest.raises(SystemExit) as ei:
        _fetch_mod.main([
            "8955",
            "--from", "2026-04-20", "--to", "2026-04-25", "--dry-run",
        ])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "sangiin\t8955" in out      # explicit target も出る (回帰防止)
    assert "shugiin\t56239" in out


# ---------- list.render ----------

def test_weekday():
    assert _weekday("2026-05-14") == "木"
    assert _weekday("2026-05-18") == "月"
    assert _weekday("invalid") == "?"


def test_render_human_empty():
    out = render_human([])
    assert "0 件" in out


def test_render_human_basic():
    items = [
        {
            "house": "sangiin", "id": "8955", "date": "2026-04-15",
            "title": "デジタル特別委員会", "duration": "5時間", "fetched": True,
            "page_url": "https://example.com/?sid=8955",
        },
        {
            "house": "shugiin", "id": "56239", "date": "2026-05-14",
            "title": "デジタル特別委員会", "duration": "2時間49分", "fetched": False,
            "page_url": "https://example.com/?deli_id=56239",
        },
    ]
    out = render_human(items)
    assert "2 件" in out
    assert "[参議院]" in out
    assert "[衆議院]" in out
    assert "✓ 取込済" in out  # sangiin の方
    # 日付順
    assert out.index("2026-04-15") < out.index("2026-05-14")


def test_render_human_missing_id_does_not_emit_empty_fetch_command():
    items = [{
        "house": "sangiin", "id": "", "date": "2026-05-14",
        "title": "内閣委員会", "duration": "", "fetched": False,
        "page_url": "https://www.webtv.sangiin.go.jp/webtv/index.php",
    }]
    out = render_human(items)
    assert "id=(未解決)" in out
    assert "kokkai fetch  " not in out


def test_render_json_payload():
    items = [{"house": "shugiin", "id": "56239", "date": "2026-05-14", "title": "テスト"}]
    raw = render_json(items)
    data = json.loads(raw)
    assert data["ok"] is True
    assert data["n_items"] == 1
    assert data["items"][0]["id"] == "56239"


def test_render_jsonl_lines():
    items = [
        {"house": "shugiin", "id": "56239", "date": "2026-05-14"},
        {"house": "shugiin", "id": "56240", "date": "2026-05-14"},
    ]
    raw = render_jsonl(items)
    lines = raw.split("\n")
    assert len(lines) == 3  # summary + 2 items
    summary = json.loads(lines[0])
    assert summary["type"] == "summary"
    assert summary["n_items"] == 2
    for line in lines[1:]:
        d = json.loads(line)
        assert d["type"] == "item"


# ---------- list.status (取込済判定) ----------

def test_load_fetched_index_empty(tmp_path: Path):
    assert load_fetched_index(tmp_path) == {}


def test_load_fetched_index_with_meta(tmp_path: Path):
    (tmp_path / "test.meta.json").write_text(
        json.dumps({
            "house": "sangiin", "id": "8955", "date": "2026-04-15",
            "title": "テスト委員会", "files": {"html": "test.html"},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    idx = load_fetched_index(tmp_path)
    assert ("sangiin", "8955") in idx
    assert idx[("sangiin", "8955")]["files"]["html"] == "test.html"


def test_annotate_marks_fetched(tmp_path: Path):
    (tmp_path / "test.meta.json").write_text(
        json.dumps({"house": "sangiin", "id": "8955", "title": "x"}, ensure_ascii=False),
        encoding="utf-8",
    )
    items = [
        {"house": "sangiin", "id": "8955", "title": "fetched one"},
        {"house": "sangiin", "id": "9999", "title": "not yet"},
    ]
    annotated = annotate(items, tmp_path)
    assert annotated[0]["fetched"] is True
    assert annotated[1]["fetched"] is False


def test_annotate_handles_corrupted_meta(tmp_path: Path):
    (tmp_path / "bad.meta.json").write_text("{not valid json", encoding="utf-8")
    idx = load_fetched_index(tmp_path)
    assert idx == {}


# ---------- list.sangiin_sid (通信失敗と存在しない sid の区別) ----------

def test_fetch_detail_transient_failure_is_not_cached(monkeypatch):
    cache = {"detail": {}, "by_date": {}, "sid_max": None, "sid_max_at": None}

    def fail(_url: str) -> str:
        raise OSError("temporary network failure")

    monkeypatch.setattr(sangiin_sid, "_http_get", fail)
    with pytest.raises(sangiin_sid.SidLookupUnavailable):
        sangiin_sid._fetch_detail(9999, cache)
    assert "9999" not in cache["detail"]


def test_resolve_sids_returns_none_without_saving_bad_cache(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "sangiin_sid_cache.json"
    monkeypatch.setattr(sangiin_sid, "_cache_path", lambda: cache_path)

    def fail(_cache: dict) -> int:
        raise sangiin_sid.SidLookupUnavailable("temporary network failure")

    monkeypatch.setattr(sangiin_sid, "_probe_sid_max", fail)
    out = sangiin_sid.resolve_sids_for_date(date(2026, 5, 14), ["内閣委員会"])
    assert out == {"内閣委員会": None}
    assert not cache_path.exists()
