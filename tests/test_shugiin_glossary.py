"""glossary.py (静的置換) のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

from kokkai.shugiin.glossary import (
    DEFAULT_PARLIAMENT_GLOSSARY,
    apply_glossary,
    load_glossary,
)


# ---------- 既定辞書 ----------

def test_default_glossary_nonempty():
    assert len(DEFAULT_PARLIAMENT_GLOSSARY) > 30


def test_default_glossary_known_pairs():
    g = DEFAULT_PARLIAMENT_GLOSSARY
    # 議事系
    assert g["前回一致"] == "全会一致"
    assert g["不対決議"] == "附帯決議"
    assert g["規律多数"] == "起立多数"
    # 法律系
    assert g["青年貢献"] == "成年後見"
    assert g["公権開始"] == "後見開始"
    # 経済系
    assert g["体内直接投資"] == "対内直接投資"


# ---------- load_glossary ----------

def test_load_glossary_defaults_only():
    g = load_glossary(None, include_defaults=True)
    assert g == DEFAULT_PARLIAMENT_GLOSSARY


def test_load_glossary_no_defaults_no_path():
    g = load_glossary(None, include_defaults=False)
    assert g == {}


def test_load_glossary_user_file_overrides_defaults(tmp_path: Path):
    path = tmp_path / "custom.txt"
    path.write_text(
        "# コメント行\n"
        "\n"
        "前回一致 → 全員一致\n"
        "新規用語 → 新規\n"
        "矢印未対応行\n",
        encoding="utf-8",
    )
    g = load_glossary(path, include_defaults=True)
    # user 定義が default を上書き
    assert g["前回一致"] == "全員一致"
    # 新規 user エントリ
    assert g["新規用語"] == "新規"
    # default の他のエントリは残る
    assert g["不対決議"] == "附帯決議"


def test_load_glossary_arrow_variants(tmp_path: Path):
    path = tmp_path / "custom.txt"
    path.write_text(
        "全角矢印 → 結果1\n"
        "半角矢印 -> 結果2\n",
        encoding="utf-8",
    )
    g = load_glossary(path, include_defaults=False)
    assert g["全角矢印"] == "結果1"
    assert g["半角矢印"] == "結果2"


# ---------- apply_glossary ----------

def test_apply_glossary_basic():
    cues = [
        {"text": "本案は前回一致をもって可決すべきものと決しました"},
        {"text": "なお不対決議が付されました"},
        {"text": "なにもしない"},
    ]
    n = apply_glossary(cues, DEFAULT_PARLIAMENT_GLOSSARY)
    assert n == 2
    assert "全会一致" in cues[0]["text"]
    assert "前回一致" not in cues[0]["text"]
    assert "附帯決議" in cues[1]["text"]
    assert cues[2]["text"] == "なにもしない"


def test_apply_glossary_longer_keys_first():
    """短いキーが長いキーの一部を破壊しないことを確認 (長いキーから順に適用)。

    例: 「補助開始の審刊」を「補助開始の審判」に直してから「審刊」→「審判」を当てれば
    安全。逆順だと「補助開始の審刊」中の「審刊」を先に直して「補助開始の審判」になるが、
    結果的に同じ。一方、glossary に「補助開始の審刊 → 後見開始の審判」のような長い
    rewrite があると順序が効く。
    """
    glossary = {
        "の審刊": "の審判",
        "補助開始の審刊": "後見開始の審判",
    }
    cues = [{"text": "補助開始の審刊により"}]
    apply_glossary(cues, glossary)
    # 長い方が先に当たって「後見開始の審判」になる (短いキーは既に消費済み)
    assert "後見開始の審判" in cues[0]["text"]


def test_apply_glossary_empty():
    assert apply_glossary([], DEFAULT_PARLIAMENT_GLOSSARY) == 0
    assert apply_glossary([{"text": "x"}], {}) == 0


def test_apply_glossary_count_per_cue():
    """同一 cue 内で複数置換があっても 1 件カウント (cue 単位)。"""
    cues = [{"text": "前回一致と不対決議"}]
    n = apply_glossary(cues, DEFAULT_PARLIAMENT_GLOSSARY)
    assert n == 1
    assert "全会一致" in cues[0]["text"]
    assert "附帯決議" in cues[0]["text"]
