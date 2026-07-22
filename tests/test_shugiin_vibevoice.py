"""vibevoice backend の結合テスト (モデルはモック、ffmpeg は実物)。

_transcribe_vibevoice のプラミング (24kHz resample → chunk 分割 → 累積 offset
→ parsed 出力の cue 変換) を、transformers 本体なしで検証する。
ffmpeg が無い環境では skip。
"""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

import kokkai.shugiin.asr as asr
from kokkai.errors import AsrError

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg が必要"
)


def _make_wav(path: Path, sec: float, rate: int = 16000) -> Path:
    """無音 PCM wav を生成 (ffmpeg 不要、標準 wave のみ)。"""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(sec * rate))
    return path


# ----- resample / chunk 分割 (実 ffmpeg) -----


@needs_ffmpeg
def test_prepare_vibevoice_wav_resamples_to_24k(tmp_path):
    wav = _make_wav(tmp_path / "in.wav", sec=1.0, rate=16000)
    out = asr._prepare_vibevoice_wav(wav)
    assert out.name == "in_24k.wav"
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == asr.VIBEVOICE_SAMPLE_RATE
        assert w.getnchannels() == 1
    # 2 回目はキャッシュ再利用 (mtime が変わらない)
    mtime = out.stat().st_mtime_ns
    assert asr._prepare_vibevoice_wav(wav) == out
    assert out.stat().st_mtime_ns == mtime


@needs_ffmpeg
def test_split_wav_chunks_short_input_is_passthrough(tmp_path):
    wav = _make_wav(tmp_path / "short.wav", sec=2.0, rate=24000)
    assert asr._split_wav_chunks(wav, chunk_sec=10) == [wav]


@needs_ffmpeg
def test_split_wav_chunks_splits_and_preserves_total_duration(tmp_path):
    wav = _make_wav(tmp_path / "long.wav", sec=10.0, rate=24000)
    chunks = asr._split_wav_chunks(wav, chunk_sec=3)
    assert len(chunks) == 4
    assert [c.name for c in chunks] == sorted(c.name for c in chunks)
    total = sum(asr._wav_duration_sec(c) for c in chunks)
    assert total == pytest.approx(10.0, abs=0.05)


# ----- _transcribe_vibevoice 全体 (transformers をモック) -----


class _FakeIds:
    """input_ids / generate 出力の代役。shape とスライスだけ模倣。"""

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, item):
        return self


class _FakeBatch(dict):
    def to(self, *args, **kwargs):
        return self


class _FakeProcessor:
    """apply_transcription_request の呼び出しを記録し、canned 出力を返す。"""

    def __init__(self, decoded_per_call):
        self.decoded_per_call = list(decoded_per_call)
        self.requests: list[dict] = []
        self._i = 0

    def apply_transcription_request(self, *, audio, **kwargs):
        self.requests.append({"audio": audio, **kwargs})
        return _FakeBatch(input_ids=_FakeIds((1, 7)))

    def decode(self, generated_ids, return_format):
        assert return_format == "parsed"
        out = self.decoded_per_call[self._i]
        self._i += 1
        return out


class _FakeModel:
    device = "cpu"
    dtype = None

    def generate(self, *, input_ids, max_new_tokens, do_sample, **kwargs):
        assert do_sample is False
        return _FakeIds((1, 12))


def _install_fakes(monkeypatch, processor, chunk_sec=None):
    monkeypatch.setattr(
        asr, "_load_vibevoice",
        lambda: (
            type("AP", (), {"from_pretrained": staticmethod(lambda mid: processor)}),
            type("VV", (), {"from_pretrained": staticmethod(
                lambda mid, **kw: _FakeModel())}),
        ),
    )
    import kokkai.shugiin._models as models
    monkeypatch.setattr(models, "ensure_model_downloaded", lambda *a, **k: None)
    if chunk_sec is not None:
        orig = asr._split_wav_chunks
        monkeypatch.setattr(
            asr, "_split_wav_chunks", lambda p: orig(p, chunk_sec=chunk_sec)
        )


@needs_ffmpeg
def test_transcribe_vibevoice_end_to_end_with_chunk_offsets(tmp_path, monkeypatch):
    """4 秒 wav を 2 秒 chunk に分割 → 各 chunk の cue に累積 offset が付く。"""
    wav = _make_wav(tmp_path / "meeting.wav", sec=4.0, rate=16000)
    processor = _FakeProcessor([
        # batch 形式 (list[list[dict]])
        [[{"Start": 0.5, "End": 1.0, "Speaker": 0, "Content": "前半"}]],
        # 単一形式 (list[dict]) も許容される
        [{"Start": 0.2, "End": 0.8, "Speaker": 1, "Content": "後半"}],
    ])
    _install_fakes(monkeypatch, processor, chunk_sec=2)

    cues = asr._transcribe_vibevoice(wav, "ヒント文", "fake/model", "ja")

    assert [c["text"] for c in cues] == ["前半", "後半"]
    assert cues[0]["start"] == pytest.approx(0.5, abs=0.01)
    # 2 個目の chunk は offset ≈ 2.0s
    assert cues[1]["start"] == pytest.approx(2.2, abs=0.05)
    assert cues[1]["end"] == pytest.approx(2.8, abs=0.05)
    # hint は prompt として全 chunk に渡る
    assert [r.get("prompt") for r in processor.requests] == ["ヒント文", "ヒント文"]
    # 入力は 24kHz 変換後の chunk ファイル
    assert all("_24k" in r["audio"] for r in processor.requests)


@needs_ffmpeg
def test_transcribe_vibevoice_without_hint_omits_prompt(tmp_path, monkeypatch):
    wav = _make_wav(tmp_path / "m.wav", sec=1.0, rate=16000)
    processor = _FakeProcessor([[[{"Start": 0.0, "End": 0.5, "Content": "a"}]]])
    _install_fakes(monkeypatch, processor)
    asr._transcribe_vibevoice(wav, None, "fake/model", "ja")
    assert "prompt" not in processor.requests[0]


@needs_ffmpeg
def test_transcribe_vibevoice_parse_failure_raises(tmp_path, monkeypatch):
    """decode が parse 失敗で生テキスト (str) を返したら AsrError。"""
    wav = _make_wav(tmp_path / "m.wav", sec=1.0, rate=16000)
    for bad in (["assistant なにか壊れた出力"], "assistant なにか壊れた出力"):
        processor = _FakeProcessor([bad])
        _install_fakes(monkeypatch, processor)
        with pytest.raises(AsrError, match="parse できませんでした"):
            asr._transcribe_vibevoice(wav, None, "fake/model", "ja")
