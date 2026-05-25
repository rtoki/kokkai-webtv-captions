"""
sherpa-onnx 話者分離 POC: 音響特徴で話者を切り分け、既存 ASR cue に speaker_id を付与する。

実行例:
  uv run python scripts/asr_poc_diarize.py out/<wav> \\
      --transcript out/<...>_transcript.json

入力:
  - wav (16kHz mono PCM16)
  - 任意で transcript.json (既存 ASR の cue 列)

処理:
  1. sherpa-onnx の OfflineSpeakerDiarization で wav を話者別に分離
     → segments = [(start, end, speaker_id), ...]
  2. transcript が与えられたら、各 cue を時刻が overlap する speaker_id に紐付け
  3. speaker_id 別の cue 数と先頭 segments を表示

出力:
  - out_poc/diarize/<stem>_segments.json: 話者分離の結果のみ
  - out_poc/diarize/<stem>_cues_with_speaker.json: cues + speaker_id

注意:
  - 初回は ~60MB のモデル DL (pyannote segmentation + 3D-Speaker embedding)
  - 話者 ID は匿名 (0/1/2/...) で、誰が誰かは別途同定が必要
  - HF token 不要 (sherpa-onnx repo 配布のモデルを使う)
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import urllib.request
import wave
from collections import Counter
from pathlib import Path


# sherpa-onnx 配布の話者分離モデル (HF token 不要)。
SEG_MODEL_TAR = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMB_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

CACHE_DIR = Path.home() / ".cache" / "sherpa-onnx-diarize"
SEG_MODEL_PATH = (
    CACHE_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
)
EMB_MODEL_PATH = (
    CACHE_DIR / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)


def _ensure_models() -> None:
    """初回起動時に DL。すでに展開済みなら skip。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not SEG_MODEL_PATH.exists():
        print(f"[poc] DL (segmentation): {SEG_MODEL_TAR}", file=sys.stderr)
        tar_path = CACHE_DIR / "seg.tar.bz2"
        urllib.request.urlretrieve(SEG_MODEL_TAR, tar_path)
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(CACHE_DIR)
        tar_path.unlink()
        print(f"[poc] 展開完了: {SEG_MODEL_PATH}", file=sys.stderr)

    if not EMB_MODEL_PATH.exists():
        print(f"[poc] DL (embedding): {EMB_MODEL_URL}", file=sys.stderr)
        urllib.request.urlretrieve(EMB_MODEL_URL, EMB_MODEL_PATH)
        print(f"[poc] DL 完了: {EMB_MODEL_PATH}", file=sys.stderr)


def _read_wav_float32(path: Path):
    """16kHz mono PCM16 wav を float32 (-1〜1) で読み、(N, 1) 形に。"""
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nf = wf.getnframes()
        if sr != 16000 or nch != 1 or sw != 2:
            raise SystemExit(f"想定外 wav: sr={sr} ch={nch} sw={sw}")
        raw = wf.readframes(nf)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # sherpa-onnx の process() は 1D float32 配列 (mono samples) を期待
    return samples, sr


def _diarize(
    wav_path: Path,
    *,
    num_speakers: int = -1,
    threshold: float = 0.5,
    min_on: float = 0.3,
    min_off: float = 0.5,
) -> tuple[list[tuple[float, float, int]], float, float]:
    """sherpa-onnx で話者分離。"""
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_MODEL_PATH),
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMB_MODEL_PATH),
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold,
        ),
        min_duration_on=min_on,
        min_duration_off=min_off,
    )

    print("[poc] sherpa-onnx OfflineSpeakerDiarization ロード...", file=sys.stderr)
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    audio, sr = _read_wav_float32(wav_path)
    duration = len(audio) / sr
    print(f"[poc] 話者分離開始: {wav_path.name} ({duration:.0f}s)", file=sys.stderr)

    t0 = time.perf_counter()
    result = sd.process(audio).sort_by_start_time()
    elapsed = time.perf_counter() - t0

    segments = [(float(r.start), float(r.end), int(r.speaker)) for r in result]
    rtf = elapsed / duration if duration > 0 else 0.0
    n_speakers = len({s[2] for s in segments})
    print(
        f"[poc] 完了: {elapsed:.1f}s / 音声 {duration:.0f}s = RTF {rtf:.3f}",
        file=sys.stderr,
    )
    print(
        f"[poc] segments={len(segments)}, 話者数={n_speakers}",
        file=sys.stderr,
    )
    return segments, elapsed, duration


def _assign_speakers_to_cues(
    cues: list[dict],
    segments: list[tuple[float, float, int]],
) -> list[dict]:
    """各 cue に最も overlap する speaker_id を付与した新 list を返す。"""
    out: list[dict] = []
    for cue in cues:
        cs = float(cue.get("start", 0.0))
        ce = float(cue.get("end", cs))
        best_overlap = 0.0
        best_speaker: int | None = None
        for ss, se, sp in segments:
            if se < cs or ss > ce:
                continue
            ov = min(ce, se) - max(cs, ss)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = sp
        new_cue = dict(cue)
        new_cue["speaker_id"] = best_speaker
        out.append(new_cue)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="sherpa-onnx 話者分離 POC")
    ap.add_argument("wav", type=Path)
    ap.add_argument(
        "--transcript", type=Path, default=None,
        help="既存 transcript.json (cue 列)。指定すると cue 毎に speaker_id を付与。",
    )
    ap.add_argument(
        "--num-speakers", type=int, default=-1,
        help="話者数 (-1 で自動推定)。事前に分かっていれば指定するとクラスタリングが安定。",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.5,
        help="クラスタリング閾値 (0.3-0.7)。小さいほど話者を細かく分ける。",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("out_poc/diarize"))
    args = ap.parse_args()

    if not args.wav.exists():
        raise SystemExit(f"wav が無い: {args.wav}")

    _ensure_models()

    segments, elapsed, duration = _diarize(
        args.wav,
        num_speakers=args.num_speakers,
        threshold=args.threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = args.out_dir / f"{args.wav.stem}_segments.json"
    seg_path.write_text(
        json.dumps(
            {
                "wav": str(args.wav),
                "duration_sec": duration,
                "elapsed_sec": elapsed,
                "rtf": elapsed / duration if duration > 0 else 0.0,
                "num_speakers": len({s[2] for s in segments}),
                "num_segments": len(segments),
                "segments": [
                    {"start": s, "end": e, "speaker": sp}
                    for s, e, sp in segments
                ],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[poc] 保存: {seg_path}", file=sys.stderr)

    if args.transcript and args.transcript.exists():
        tr = json.loads(args.transcript.read_text(encoding="utf-8"))
        cues = tr.get("cues", [])
        new_cues = _assign_speakers_to_cues(cues, segments)
        cue_path = args.out_dir / f"{args.wav.stem}_cues_with_speaker.json"
        cue_path.write_text(
            json.dumps(
                {
                    "wav": str(args.wav),
                    "num_cues": len(new_cues),
                    "cues": new_cues,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[poc] 保存: {cue_path}", file=sys.stderr)

        speakers = Counter(c.get("speaker_id") for c in new_cues)
        chars_by_speaker = Counter()
        for c in new_cues:
            chars_by_speaker[c.get("speaker_id")] += len(c.get("text", ""))
        print("\n--- speaker_id 別 cue 数 / 文字数 ---")
        for sp, n_cue in speakers.most_common():
            n_char = chars_by_speaker[sp]
            label = f"speaker_{sp:02d}" if sp is not None else "(unassigned)"
            print(f"  {label}  cue={n_cue:>4}  chars={n_char:>5}")

    print(f"\n--- 先頭 15 segments ---")
    for s, e, sp in segments[:15]:
        mm, ss_ = int(s // 60), int(s % 60)
        em, es_ = int(e // 60), int(e % 60)
        print(
            f"  speaker_{sp:02d}  "
            f"[{mm:3d}:{ss_:02d} - {em:3d}:{es_:02d}]  ({e-s:.1f}s)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
