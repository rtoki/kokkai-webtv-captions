"""llm_context.py の純関数 (LLM 推論不要部分) のテスト。"""

from __future__ import annotations

from kokkai.shugiin.llm_context import _parse_terms, merge_into_prompt


# ---------- _parse_terms ----------


def test_parse_terms_object_with_terms_key():
    raw = '{"terms": ["AISI", "外為法", "もんじゅ"]}'
    assert _parse_terms(raw) == ["AISI", "外為法", "もんじゅ"]


def test_parse_terms_bare_array():
    raw = '["AISI", "NISC"]'
    assert _parse_terms(raw) == ["AISI", "NISC"]


def test_parse_terms_truncates_to_max():
    # MAX_TERMS=10 を越える入力は 10 個に切る
    raw = "[" + ", ".join(f'"t{i}"' for i in range(20)) + "]"
    out = _parse_terms(raw)
    assert len(out) == 10
    assert out[0] == "t0" and out[9] == "t9"


def test_parse_terms_filters_non_string_entries():
    raw = '{"terms": ["AISI", 42, null, "外為法", ""]}'
    assert _parse_terms(raw) == ["AISI", "外為法"]


def test_parse_terms_codefence_stripped():
    raw = '```json\n{"terms": ["x"]}\n```'
    assert _parse_terms(raw) == ["x"]


def test_parse_terms_invalid_json_returns_empty(capsys):
    assert _parse_terms("not json") == []
    err = capsys.readouterr().err
    assert "WARN" in err


# ---------- merge_into_prompt ----------


def test_merge_into_prompt_empty_terms_returns_base():
    assert merge_into_prompt("会議: 法務委員会", []) == "会議: 法務委員会"
    assert merge_into_prompt(None, []) == ""


def test_merge_into_prompt_appends_new_terms():
    out = merge_into_prompt("会議: 内閣委員会", ["外為法", "もんじゅ"])
    assert "会議: 内閣委員会" in out
    assert "外為法" in out and "もんじゅ" in out


def test_merge_into_prompt_dedupes_against_base():
    base = "外為法と関連する議論"
    out = merge_into_prompt(base, ["外為法", "AISI"])
    # 既出の「外為法」は再追加しない
    assert out.count("外為法") == 1
    assert "AISI" in out


def test_merge_into_prompt_caps_length():
    base = "あ" * 500
    out = merge_into_prompt(base, ["新規語"] + ["他語"] * 10)
    assert len(out) <= 600
