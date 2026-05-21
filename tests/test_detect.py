"""発言者検出 (detect.py) のテスト."""

from __future__ import annotations

import pytest

from kokkai.sangiin.detect import (
    _chunk_by_sentence,
    classify_role_str,
    detect_speaker_calls,
    detect_turns,
)


# ---------- 単発の呼びかけパターン ----------

@pytest.mark.parametrize(
    "text,expected_name,expected_role_contains",
    [
        # 大臣 (シンプル)
        ("山田大臣まずですね、お答えします。", "山田", "大臣"),
        # ポートフォリオ名+大臣
        ("山田デジタル大臣まずですね、お答えします。", "山田", "大臣"),
        # 政府参考人 (統括監)
        ("三浦統括監はい、お答えいたします。", "三浦", "統括監"),
        # 政府参考人 (審議官)
        ("阿部審議官お答えいたします。", "阿部", "審議官"),
        # 政府参考人 (官房長官)
        ("木原官房長官お答えします。", "木原", "官房長官"),
        # 最高裁長官 (司法)
        ("山口最高裁長官お答えします。", "山口", "最高裁長官"),
        # 委員長
        ("田中委員長申し上げます。", "田中", "委員長"),
        # 議長 (本会議)
        ("田中議長申し上げます。", "田中", "議長"),
        # 副大臣
        ("佐藤副大臣お願いします。", "佐藤", "副大臣"),
        # 政務官
        ("鈴木政務官お答えします。", "鈴木", "政務官"),
        # 政府参考人 (明示)
        ("中原政府参考人お答えします。", "中原", "政府参考人"),
        # 統合幕僚長
        ("山田統合幕僚長申し上げます。", "山田", "統合幕僚長"),
        # 理事 (議院運営)
        ("山田理事はい、お答えします。", "山田", "理事"),
    ],
    ids=lambda x: str(x)[:30],
)
def test_simple_call(text, expected_name, expected_role_contains):
    """単純な『名前+役職 + 答弁開始』パターンが検出される。"""
    calls = detect_speaker_calls(text)
    assert len(calls) >= 1, f"call が検出されない: {text}"
    matched = [c for c in calls if c["name"] == expected_name]
    assert matched, f"name={expected_name} が検出されない: {calls}"
    assert expected_role_contains in matched[0]["role_str"], \
        f"role が {expected_role_contains} を含まない: {matched[0]}"


# ---------- リファレンス (助詞付き) はターン交代としない ----------

@pytest.mark.parametrize(
    "text",
    [
        "山田大臣に伺います。",        # 〜に
        "山田大臣のお考えは。",         # 〜の
        "山田大臣は素晴らしい人だ。",   # 〜は
        "山田大臣が答弁しました。",     # 〜が
        "山田大臣を呼びました。",       # 〜を
        "三浦統括監への質問です。",     # 〜へ
        "山本議長を選任します。",         # 〜を (人事のため呼びかけではない)
        "中原局長は答弁しました。",       # 〜は
    ],
    ids=lambda x: x[:25],
)
def test_reference_not_call(text):
    """役職直後が助詞なら、ターン交代ではなく『言及』として扱う。"""
    calls = detect_speaker_calls(text)
    # ターン扱いの call は 0 (助詞でブロックされる)
    assert len(calls) == 0, f"言及がターンとして拾われた: {calls}"


# ---------- 照応・指示語の誤検出を防ぐ ----------

@pytest.mark.parametrize(
    "text",
    [
        "今の大臣の答弁ありがとうございます。",
        "先日の大臣の発言は",
        "先程の大臣がおっしゃった",
        "次の大臣にお伺いします",
    ],
    ids=lambda x: x[:30],
)
def test_stoplist_no_call(text):
    """『今の大臣』『先日の大臣』など照応表現は検出されない。"""
    calls = detect_speaker_calls(text)
    # 「今」「先」等は固有名詞でないので拾われない or 助詞で弾かれる
    bad = [c for c in calls if c["name"] in {"今", "先", "次", "先日", "先程"}]
    assert not bad, f"照応が拾われた: {bad}"


# ---------- 質問者復帰 ----------

@pytest.mark.parametrize(
    "text",
    [
        "鈴木花子くん。ありがとうございました。",
        "高橋桃子君次の質問です。",
    ],
    ids=lambda x: x[:25],
)
def test_questioner_return(text):
    """『○○くん』『○○君』は質問者復帰として _return ロールを持つ。"""
    calls = detect_speaker_calls(text)
    returns = [c for c in calls if c["role_str"] == "_return"]
    assert returns, f"_return が検出されない: {calls}"


# ---------- 所属プレフィックスがあるケース ----------

def test_org_prefix_ignored_for_name():
    """『厚生労働省大臣官房榊原審議官』は name=榊原 / role=審議官 が拾える。"""
    text = "厚生労働省大臣官房榊原審議官お答えいたします。"
    calls = detect_speaker_calls(text)
    matched = [c for c in calls if c["name"] == "榊原" and "審議官" in c["role_str"]]
    assert matched, f"榊原 が拾えない: {calls}"


def test_long_title_compound():
    """『総括審議官』のような複合役職が拾える。"""
    text = "山田総括審議官にお答えいただきます。山田総括審議官はい、お答えします。"
    calls = detect_speaker_calls(text)
    # 2回出現するが、1回目は助詞 (に) で除外、2回目は本回答
    answers = [c for c in calls if c["name"] == "山田"]
    assert answers, f"総括審議官 が拾えない: {calls}"


# ---------- 分類 ----------

@pytest.mark.parametrize(
    "role_str,expected_klass",
    [
        ("デジタル大臣", "minister"),
        ("総理大臣", "minister"),
        ("副大臣", "minister"),
        ("大臣政務官", "minister"),
        ("政府参考人", "bureaucrat"),
        ("統括監", "bureaucrat"),
        ("審議官", "bureaucrat"),
        ("局長", "bureaucrat"),
        ("官房長官", "bureaucrat"),
        ("統合幕僚長", "bureaucrat"),
        ("委員長", "chair"),
        ("議長", "chair"),
        ("理事", "chair"),
        ("_return", "questioner"),
    ],
)
def test_classify_role(role_str, expected_klass):
    assert classify_role_str(role_str) == expected_klass


# ---------- detect_turns のセクション分割 ----------

def test_detect_turns_basic():
    """1議員のセクションを質問者+大臣+質問者に分割。"""
    text = "鈴木花子です。質問します。山田大臣まずですね、お答えします。"
    turns = detect_turns(text, primary_name="鈴木花子")
    assert len(turns) >= 2
    # 1番目は若井 (questioner)
    assert turns[0]["klass"] == "questioner"
    assert "質問します" in turns[0]["text"]
    # 後続に松本 (minister) が含まれる
    minister_turns = [t for t in turns if t["klass"] == "minister"]
    assert minister_turns
    assert any("山田" in t["speaker"] for t in minister_turns)


def test_detect_turns_no_calls():
    """セクション内に役職呼びかけが無ければ、全文を questioner として返す。"""
    text = "鈴木花子です。今日はAIについて質問させていただきます。"
    turns = detect_turns(text, primary_name="鈴木花子")
    assert len(turns) == 1
    assert turns[0]["klass"] == "questioner"
    assert turns[0]["speaker"] == "鈴木花子"


def test_detect_turns_consecutive_merge():
    """同一発言者の連続セグメントはマージされる。"""
    text = "鈴木花子です。山田大臣お答えします。山田大臣続けて申し上げます。"
    turns = detect_turns(text, primary_name="鈴木花子")
    minister_turns = [t for t in turns if t["klass"] == "minister"]
    assert len(minister_turns) == 1, f"連続が統合されていない: {minister_turns}"


# ---------- SudachiPy 入力長対策 ----------


def test_chunk_by_sentence_short_text_returns_single():
    text = "短い文章です。"
    assert _chunk_by_sentence(text, max_bytes=1000) == [(0, text)]


def test_chunk_by_sentence_splits_on_period():
    # 各文は 30 byte 前後。max_bytes=60 で 2-3 文ごとに区切る想定。
    s1 = "これは一つ目の文です。"  # ~30 byte
    s2 = "二つ目の文です。"        # ~24 byte
    s3 = "三つ目の文ですね。"      # ~27 byte
    text = s1 + s2 + s3
    chunks = _chunk_by_sentence(text, max_bytes=40)
    # 文字数の合計が原文と一致 (チャンク間でロスなし)
    assert "".join(c[1] for c in chunks) == text
    # 1 チャンクが max_bytes より小さい (文末で割れたため超過しないことを保証)
    for _, piece in chunks:
        assert len(piece.encode("utf-8")) <= 40 or piece == text
    # 少なくとも 2 チャンクに割れる
    assert len(chunks) >= 2


def test_chunk_by_sentence_offsets_are_cumulative():
    text = "A大臣が答えました。B大臣が答えました。" * 100  # ~50 文 ≈ 1500+ char
    chunks = _chunk_by_sentence(text, max_bytes=200)
    # 各チャンクの start_offset = 前チャンクの累積長
    cumulative = 0
    for start, piece in chunks:
        assert start == cumulative
        cumulative += len(piece)
    assert cumulative == len(text)


def test_detect_speaker_calls_handles_long_text():
    """50KB 超の text (SudachiPy 単体だと InputTooLong) でも例外を投げず動く。"""
    # 役職直後に助詞を置かないパターン (cf. test_simple_call) を 2000 回繰り返し
    # 50KB 超になる。SudachiPy の 49149 byte 上限を超えるので chunking が必須。
    repeat = "山田大臣お答えします。" * 2000
    assert len(repeat.encode("utf-8")) > 60_000
    calls = detect_speaker_calls(repeat)
    # 例外なく call を抽出できる
    assert len(calls) > 100
    assert all(c["name"] == "山田" for c in calls)
