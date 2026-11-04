"""VibeVoice-ASR (mlx-audio) を長尺 wav にかける subprocess 用ランナー。

``asr._transcribe_vibevoice`` から ``python -m kokkai.shugiin.vibevoice_runner`` で
起動される。別プロセスにしているのは、ASR 後に LLM 校正 (Qwen 9B) をロードする
際に VibeVoice の重み (4bit で ~5.7GB) をメモリから確実に解放するため
(16GB Mac で両方が同居すると swap する)。

VibeVoice-ASR は 1 パス最大 59 分で、それを超えると黙って切り捨てる。
加えて長尺ほど activation メモリが膨らむので、wav を ``--chunk-sec`` (既定 600 秒)
ごとに区切って順に流す。区切り位置は目標時刻の手前 60 秒の中で
最も音量の小さい 0.5 秒窓を選び、発話の途中で切らないようにする。

出力 JSON: ``{"cues": [{"start", "end", "text"}, ...], "chunks": [...]}``。
cue の時刻は元 wav 上の絶対秒。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "mlx-community/VibeVoice-ASR-4bit"

# 区切り位置を探す窓 (目標時刻の手前 SEARCH_SEC 秒) と RMS 計算の粒度。
_SEARCH_SEC = 60.0
_FRAME_SEC = 0.5

# 完全な JSON object 1 個ぶん (max_tokens で末尾が切れたときの救済用)。
_OBJ_RE = re.compile(r"\{[^{}]*\}")


def _quietest_cut(audio, sr: int, lo: float, hi: float) -> float:
    """[lo, hi) 秒の範囲で最も RMS の小さい 0.5 秒窓の中心時刻を返す。"""
    import numpy as np

    frame = int(_FRAME_SEC * sr)
    a = int(lo * sr)
    b = int(hi * sr)
    seg = audio[a:b]
    n = len(seg) // frame
    if n == 0:
        return hi
    rms = np.sqrt((seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    i = int(rms.argmin())
    return lo + (i + 0.5) * _FRAME_SEC


def _plan_chunks(audio, sr: int, start: float, end: float, chunk_sec: float) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    t = start
    while t < end - 1.0:
        target = t + chunk_sec
        if target >= end:
            cut = end
        else:
            cut = _quietest_cut(audio, sr, max(t + 1.0, target - _SEARCH_SEC), target)
        chunks.append((t, cut))
        t = cut
    return chunks


def _to_sec(v) -> float | None:
    """VibeVoice の時刻 (数値 / "12.3" / "00:01:02.3") を秒に。"""
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    try:
        sec = 0.0
        for p in parts:
            sec = sec * 60 + float(p)
        return sec
    except ValueError:
        return None


def _salvage(text: str) -> list[dict]:
    """JSON 全体のパースに失敗したとき、完全な object だけ拾う。"""
    out = []
    for m in _OBJ_RE.finditer(text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        out.append({
            "start": d.get("Start time", d.get("Start")),
            "end": d.get("End time", d.get("End")),
            "text": d.get("Content", ""),
        })
    return out


_SENT_SPLIT_RE = re.compile(r"(?<=[。！？\n])")


def _split_sentences(st: float, en: float, text: str) -> list[dict]:
    """~30 秒の VibeVoice segment を句点で文に割り、時刻を文字数比で按分する。

    発言者割付 (assign_cues_to_speakers) は cue 単位なので、委員長→質問者の
    交代を 1 cue に含んだままだと境界がずれる。按分は近似だが文単位にはなる。
    """
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        return [{"start": st, "end": en, "text": text.strip()}]
    total = sum(len(p) for p in parts)
    out, t = [], st
    for p in parts:
        nxt = t + (en - st) * len(p) / total
        out.append({"start": round(t, 2), "end": round(nxt, 2), "text": p})
        t = nxt
    out[-1]["end"] = round(en, 2)
    return out


def _segments_to_cues(segments: list[dict], offset: float) -> list[dict]:
    cues = []
    for s in segments:
        st = _to_sec(s.get("start"))
        en = _to_sec(s.get("end"))
        text = str(s.get("text") or "").strip()
        if st is None or en is None or not text:
            continue
        cues.extend(_split_sentences(st + offset, en + offset, text))
    return cues


def run(
    wav_path: Path,
    out_path: Path,
    *,
    model_id: str,
    context: str | None,
    clip_start: float,
    clip_end: float | None,
    chunk_sec: float,
    max_tokens: int,
) -> None:
    import mlx.core as mx
    import soundfile as sf
    from mlx_audio.stt.utils import load_model

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    end = min(clip_end, duration) if clip_end is not None else duration
    chunks = _plan_chunks(audio, sr, clip_start, end, chunk_sec)

    print(f"[asr] VibeVoice ロード中: {model_id}", file=sys.stderr)
    model = load_model(model_id)
    print(
        f"[asr] 文字起こし開始: {wav_path.name} ({end - clip_start:.0f}s を {len(chunks)} 分割)",
        file=sys.stderr,
    )

    cues: list[dict] = []
    chunk_log: list[dict] = []
    for i, (a, b) in enumerate(chunks, 1):
        t0 = time.time()
        seg_audio = audio[int(a * sr): int(b * sr)]
        out = model.generate(seg_audio, sampling_rate=sr, context=context, max_tokens=max_tokens)
        segs = out.segments or _salvage(out.text)
        got = _segments_to_cues(segs, a)
        truncated = out.generation_tokens >= max_tokens
        cues.extend(got)
        el = time.time() - t0
        peak_gb = mx.get_peak_memory() / 1e9
        chunk_log.append({
            "start": round(a, 2), "end": round(b, 2), "cues": len(got),
            "tokens": out.generation_tokens, "truncated": truncated, "sec": round(el, 1),
            "peak_gb": round(peak_gb, 2),
        })
        warn = " ⚠ max_tokens 到達 (末尾欠落の可能性)" if truncated else ""
        print(
            f"[asr]   chunk {i}/{len(chunks)} {a:.0f}-{b:.0f}s: cue {len(got)}"
            f" / {el:.0f}s (RTF {el / max(b - a, 1):.2f}, peak {peak_gb:.1f}GB){warn}",
            file=sys.stderr,
        )

    cues.sort(key=lambda c: c["start"])
    out_path.write_text(
        json.dumps({"model": model_id, "cues": cues, "chunks": chunk_log}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[asr] 完了 (vibevoice): cue数={len(cues)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vibevoice_runner")
    p.add_argument("wav", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--context", default=None, help="hotwords (initial_prompt 相当)")
    p.add_argument("--clip-start", type=float, default=0.0)
    p.add_argument("--clip-end", type=float, default=None)
    p.add_argument("--chunk-sec", type=float, default=600.0)
    p.add_argument("--max-tokens", type=int, default=16384)
    a = p.parse_args(argv)
    run(
        a.wav, a.out,
        model_id=a.model, context=a.context,
        clip_start=a.clip_start, clip_end=a.clip_end,
        chunk_sec=a.chunk_sec, max_tokens=a.max_tokens,
    )


if __name__ == "__main__":
    main()
