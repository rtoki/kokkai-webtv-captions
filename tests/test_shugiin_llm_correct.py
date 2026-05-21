"""llm_correct.py のうちネット (HTTP) 不要な純関数のテスト。

実際の vllm-mlx 呼び出しは E2E で別途検証。
"""

from __future__ import annotations

from kokkai.shugiin.llm_correct import (
    _build_user_message,
    _is_hallucination_text,
    _is_repetitive_loop,
    _parse_corrections,
    _strip_inline_loops,
    default_model_for,
    drop_hallucinations,
    preclean_loops,
)


# ---------- _build_user_message ----------

def test_build_user_message_basic():
    cues = [
        {"id": 0, "text": "テキスト1"},
        {"id": 1, "text": "テキスト2"},
    ]
    msg = _build_user_message(cues, context_hint="法務委員会")
    import json
    data = json.loads(msg)
    assert "cues" in data
    assert "context" in data
    assert data["context"] == "法務委員会"
    assert len(data["cues"]) == 2
    assert data["cues"][0]["text"] == "テキスト1"


def test_build_user_message_no_hint():
    msg = _build_user_message([{"id": 0, "text": "x"}], context_hint=None)
    import json
    data = json.loads(msg)
    assert "context" not in data
    assert data["cues"] == [{"id": 0, "text": "x"}]


def test_build_user_message_assigns_id_from_index_when_missing():
    cues = [{"text": "a"}, {"text": "b"}]
    msg = _build_user_message(cues, context_hint=None)
    import json
    data = json.loads(msg)
    assert [c["id"] for c in data["cues"]] == [0, 1]


# ---------- _parse_corrections ----------

def test_parse_corrections_wrapped_object():
    """vllm-mlx の json_object モードは ``{"corrections": [...]}`` を返すのが典型。"""
    raw = '{"corrections": [{"id": 0, "text": "修正後"}, {"id": 2, "text": "別の修正"}]}'
    out = _parse_corrections(raw)
    assert len(out) == 2
    assert out[0] == {"id": 0, "text": "修正後"}
    assert out[1]["id"] == 2


def test_parse_corrections_bare_array():
    """念のため [...] が直接来た場合も受ける。"""
    raw = '[{"id": 1, "text": "x"}]'
    out = _parse_corrections(raw)
    assert out == [{"id": 1, "text": "x"}]


def test_parse_corrections_codefence_stripped():
    raw = "```json\n" + '{"corrections": [{"id": 0, "text": "y"}]}' + "\n```"
    out = _parse_corrections(raw)
    assert out == [{"id": 0, "text": "y"}]


def test_parse_corrections_invalid_json_returns_empty(capsys):
    out = _parse_corrections("not json at all")
    assert out == []
    err = capsys.readouterr().err
    assert "WARN" in err


def test_parse_corrections_filters_malformed_entries():
    raw = '{"corrections": [{"id": 0, "text": "ok"}, {"id": 1}, "not a dict", {"text": "no id"}]}'
    out = _parse_corrections(raw)
    assert len(out) == 1
    assert out[0] == {"id": 0, "text": "ok"}


def test_parse_corrections_empty_corrections_field():
    raw = '{"corrections": []}'
    assert _parse_corrections(raw) == []


# ---------- preclean_loops / _is_repetitive_loop ----------


def test_is_repetitive_loop_short_text_is_safe():
    assert not _is_repetitive_loop("ティティ")
    assert not _is_repetitive_loop("これは普通の日本語文章であります。")


def test_is_repetitive_loop_long_unique_text_is_safe():
    # 通常の日本語文 (ユニーク率 > 5%) は loop 扱いしない
    text = "本日は議題に関する重要な質疑応答が行われました。" * 10
    assert not _is_repetitive_loop(text)


def test_is_repetitive_loop_degenerate_pattern_detected():
    assert _is_repetitive_loop("ティ" * 200)
    assert _is_repetitive_loop("はいはい" * 200)


def test_preclean_loops_replaces_and_stashes_original():
    cues = [
        {"id": 0, "text": "正常なテキストです。"},
        {"id": 1, "text": "ティ" * 200},
        {"id": 2, "text": "別の正常テキスト。"},
    ]
    n = preclean_loops(cues)
    assert n == 1
    assert cues[0]["text"] == "正常なテキストです。"
    assert cues[1]["text"] == "[音声不明瞭]"
    assert cues[1]["_original_text"].startswith("ティ")
    assert cues[2]["text"] == "別の正常テキスト。"


def test_preclean_loops_empty_input():
    assert preclean_loops([]) == 0


def test_strip_inline_loops_replaces_repeating_syllable():
    """通常文中に 1-4 文字音節が 10 回以上連続反復するインライン loop を検出。"""
    text = "AIセーフティ・インシ" + ("ティ" * 50) + "他社との差別化はできず"
    cleaned, n = _strip_inline_loops(text)
    assert n == 1
    assert "[音声不明瞭]" in cleaned
    assert "ティティティ" not in cleaned
    # 前後の正常文は保持される
    assert cleaned.startswith("AIセーフティ・インシ")
    assert cleaned.endswith("他社との差別化はできず")


def test_strip_inline_loops_safe_for_short_repeats():
    """通常の日本語 (短い繰り返しはあり得る) は誤検出しない。"""
    text = "そうですそうです、おっしゃるとおりです。"
    cleaned, n = _strip_inline_loops(text)
    assert n == 0
    assert cleaned == text


def test_strip_inline_loops_no_repetition():
    """通常の文章は素通し。"""
    cleaned, n = _strip_inline_loops("これは普通の日本語の文章です。")
    assert n == 0
    assert cleaned == "これは普通の日本語の文章です。"


def test_preclean_loops_handles_inline_repetition():
    """インライン repetition も preclean_loops が捕捉して件数に含む。"""
    cues = [
        {"id": 0, "text": "正常な発言です。"},
        {
            "id": 1,
            "text": "AIセーフティ・インシ" + "ティ" * 50 + "他社との差別化",
        },
        {"id": 2, "text": "ティ" * 200},  # 全体 loop (既存パス)
    ]
    n = preclean_loops(cues)
    assert n == 2
    assert cues[0]["text"] == "正常な発言です。"
    assert "[音声不明瞭]" in cues[1]["text"]
    assert "AIセーフティ" in cues[1]["text"]
    assert cues[1]["_original_text"].count("ティ") > 40
    assert cues[2]["text"] == "[音声不明瞭]"


# ---------- drop_hallucinations ----------


def test_is_hallucination_text_youtube_phrase():
    """YouTube 終了テロップ系の典型フレーズは句点ありなしどちらも検出する。"""
    assert _is_hallucination_text("ご視聴ありがとうございました")
    assert _is_hallucination_text("ご視聴ありがとうございました。")
    assert _is_hallucination_text("ご視聴ありがとうございました ")
    assert _is_hallucination_text("ご清聴ありがとうございました")
    assert _is_hallucination_text("チャンネル登録お願いします")


def test_is_hallucination_text_repeated_phrase():
    """mlx-whisper が 1 segment に幻覚句を詰めるパターンを検出。"""
    assert _is_hallucination_text("ご視聴ありがとうございました。" * 3)
    assert _is_hallucination_text("ご視聴ありがとうございました" * 5)


def test_is_hallucination_text_real_japanese_is_safe():
    """通常の議会発言は誤検出しない。"""
    assert not _is_hallucination_text("これより会議を開きます。")
    assert not _is_hallucination_text("お答えいたします。")
    assert not _is_hallucination_text("")
    assert not _is_hallucination_text("。")


def test_drop_hallucinations_removes_pre_meeting_silence_cues():
    """開会前無音区間の幻覚 cue (本物の発言ではない) を全部捨てる。"""
    cues = [
        {"start": 0.0, "end": 30.0, "text": "ご視聴ありがとうございました。"},
        {"start": 30.0, "end": 60.0, "text": "ご視聴ありがとうございました"},
        {"start": 60.0, "end": 90.0, "text": "ご視聴ありがとうございました"},
        {"start": 1320.0, "end": 1330.0, "text": "これより会議を開きます。"},
        {"start": 1330.0, "end": 1340.0, "text": "本日の議題はです。"},
    ]
    n = drop_hallucinations(cues)
    assert n == 3
    assert [c["text"] for c in cues] == [
        "これより会議を開きます。",
        "本日の議題はです。",
    ]


def test_drop_hallucinations_empty_input():
    assert drop_hallucinations([]) == 0


# ---------- default_model_for ----------


def test_default_model_for_known_backends():
    assert "Qwen" in default_model_for("mlx") or "qwen" in default_model_for("mlx").lower()
    # openai backend は MLX モデル名を共用する (ローカル MLX サーバ前提)
    assert default_model_for("openai") == default_model_for("mlx")
    # 未知 backend は mlx 既定にフォールバック
    assert default_model_for("unknown") == default_model_for("mlx")
