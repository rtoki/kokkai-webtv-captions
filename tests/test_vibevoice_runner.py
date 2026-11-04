"""vibevoice_runner の純粋ロジック (分割計画・時刻パース・救済パース)。モデルは使わない。"""

import numpy as np

from kokkai.shugiin.vibevoice_runner import _plan_chunks, _salvage, _segments_to_cues, _to_sec

SR = 16000


def _audio_with_silence_at(total_sec: float, silent_at: list[float]) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(int(total_sec * SR)) * 0.1).astype("float32")
    for t in silent_at:
        a[int(t * SR): int((t + 1.0) * SR)] = 0.0
    return a


def test_chunks_cover_range_contiguously_and_respect_max():
    a = _audio_with_silence_at(2500, [570, 1150, 1790])
    chunks = _plan_chunks(a, SR, 0.0, 2500.0, 600.0)
    assert chunks[0][0] == 0.0 and chunks[-1][1] == 2500.0
    for (s1, e1), (s2, _) in zip(chunks, chunks[1:]):
        assert e1 == s2
    assert all(e - s <= 600.0 for s, e in chunks)


def test_chunk_cut_lands_in_silence():
    a = _audio_with_silence_at(1300, [570])
    (_, cut), *_ = _plan_chunks(a, SR, 0.0, 1300.0, 600.0)
    assert 570.0 <= cut <= 571.0


def test_short_audio_is_single_chunk_with_clip():
    a = _audio_with_silence_at(300, [])
    assert _plan_chunks(a, SR, 20.0, 250.0, 600.0) == [(20.0, 250.0)]


def test_to_sec_formats():
    assert _to_sec(12.5) == 12.5
    assert _to_sec("3.25") == 3.25
    assert _to_sec("00:01:02.5") == 62.5
    assert _to_sec("garbage") is None


def test_segments_offset_and_drop_empty():
    segs = [{"start": "1.0", "end": 2.0, "text": " 質疑 "}, {"start": 3, "end": 4, "text": ""}]
    assert _segments_to_cues(segs, 600.0) == [{"start": 601.0, "end": 602.0, "text": "質疑"}]


def test_long_segment_split_into_sentences_with_proportional_time():
    segs = [{"start": 0.0, "end": 30.0, "text": "これより会議を開きます。新藤義孝君。"}]
    cues = _segments_to_cues(segs, 0.0)
    assert [c["text"] for c in cues] == ["これより会議を開きます。", "新藤義孝君。"]
    assert cues[0]["start"] == 0.0 and cues[-1]["end"] == 30.0
    assert cues[0]["end"] == cues[1]["start"] == 20.0  # 12 字 : 6 字


def test_salvage_truncated_json():
    text = '[{"Start time": 0.0, "End time": 1.5, "Speaker ID": 0, "Content": "開会します"}, {"Start time": 1.6, "End'
    assert _salvage(text) == [{"start": 0.0, "end": 1.5, "text": "開会します"}]
