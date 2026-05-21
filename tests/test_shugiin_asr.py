"""asr.py のうちネット・モデル不要な純関数 (assign_cues_to_speakers) のテスト。

faster-whisper が必要な ``transcribe`` 関数自体のテストは E2E スモークで
別途実行する想定で、ここでは扱わない。
"""

from __future__ import annotations

import pytest

from kokkai.shugiin.asr import (
    ASR_BACKENDS,
    DEFAULT_ASR_BACKEND,
    DEFAULT_MLX_MODEL,
    assign_cues_to_speakers,
)


def test_backend_constants_are_stable():
    """ASR backend 名は CLI / docs に出ているので固定する。"""
    assert ASR_BACKENDS == ("faster", "mlx")
    assert DEFAULT_ASR_BACKEND == "faster"
    assert DEFAULT_MLX_MODEL.startswith("mlx-community/whisper-")


def test_unknown_backend_raises_asr_error():
    """``backend`` の typo は AsrError として fail-fast。"""
    from pathlib import Path

    from kokkai.errors import AsrError
    from kokkai.shugiin.asr import transcribe

    with pytest.raises(AsrError):
        transcribe(Path("/nonexistent.wav"), backend="invalid")


def test_recommended_asr_backend_apple_silicon(monkeypatch):
    """Apple Silicon + mlx_whisper 利用可なら "mlx" を推奨。"""
    import kokkai.shugiin.asr as asr

    monkeypatch.setattr(asr, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(asr, "_mlx_whisper_available", lambda: True)
    assert asr.recommended_asr_backend() == "mlx"


def test_recommended_asr_backend_falls_back_to_faster(monkeypatch):
    """非 Apple Silicon、または mlx_whisper 不可なら faster。"""
    import kokkai.shugiin.asr as asr

    monkeypatch.setattr(asr, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(asr, "_mlx_whisper_available", lambda: True)
    assert asr.recommended_asr_backend() == "faster"

    monkeypatch.setattr(asr, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(asr, "_mlx_whisper_available", lambda: False)
    assert asr.recommended_asr_backend() == "faster"


SPEAKERS = [
    {"start": 100.0, "name": "A", "group": "委員長"},
    {"start": 200.0, "name": "B", "group": "Q党"},
    {"start": 500.0, "name": "C", "group": "R党"},
]


def test_assigns_cues_within_range():
    cues = [
        {"start": 120.0, "end": 122.0, "text": "a の発言"},
        {"start": 250.0, "end": 255.0, "text": "b の発言"},
        {"start": 700.0, "end": 705.0, "text": "c の発言"},
    ]
    groups = assign_cues_to_speakers(cues, SPEAKERS)
    assert [g["name"] for g in groups] == ["A", "B", "C"]
    assert [len(g["cues"]) for g in groups] == [1, 1, 1]
    assert groups[0]["cues"][0]["text"] == "a の発言"
    assert groups[2]["cues"][0]["text"] == "c の発言"


def test_cues_before_first_speaker_go_to_first():
    """発言者リストの最初の start より前の cue (冒頭) は先頭発言者に寄せる。"""
    cues = [
        {"start": 50.0, "end": 55.0, "text": "冒頭"},
        {"start": 120.0, "end": 125.0, "text": "Aの話"},
    ]
    groups = assign_cues_to_speakers(cues, SPEAKERS)
    assert len(groups[0]["cues"]) == 2


def test_last_speaker_gets_all_tail_cues():
    cues = [
        {"start": 600.0, "end": 610.0, "text": "C前半"},
        {"start": 9999.0, "end": 10000.0, "text": "C末尾"},
    ]
    groups = assign_cues_to_speakers(cues, SPEAKERS)
    assert len(groups[2]["cues"]) == 2


def test_speakers_unsorted_input_is_sorted():
    unsorted = list(reversed(SPEAKERS))
    cues = [{"start": 120.0, "end": 122.0, "text": "x"}]
    groups = assign_cues_to_speakers(cues, unsorted)
    # 出力は start 昇順
    starts = [g["start"] for g in groups]
    assert starts == sorted(starts)


def test_empty_speakers_returns_fallback_group():
    cues = [{"start": 0.0, "end": 1.0, "text": "x"}]
    groups = assign_cues_to_speakers(cues, [])
    assert len(groups) == 1
    assert groups[0]["cues"] == cues


def test_empty_cues():
    groups = assign_cues_to_speakers([], SPEAKERS)
    assert [len(g["cues"]) for g in groups] == [0, 0, 0]


def test_boundary_cue_belongs_to_later_speaker():
    """cue.start == speakers[i+1].start ちょうど。i+1 に帰属する。"""
    cues = [{"start": 200.0, "end": 205.0, "text": "境界"}]
    groups = assign_cues_to_speakers(cues, SPEAKERS)
    # 200.0 は B (start=200) に入る
    assert len(groups[1]["cues"]) == 1
    assert len(groups[0]["cues"]) == 0
