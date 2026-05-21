"""``--skip-asr`` (transcript.json 再利用) のテスト。

実際の Whisper を回さずに `_run_phase3` の skip ブランチだけを検証する。
extract / audio / asr のネットワーク・モデル依存をモック化。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from kokkai.errors import InvalidInputError
from kokkai.shugiin import __main__ as shugiin_main


def _make_args(target: str, output: Path, **overrides) -> argparse.Namespace:
    """`_run_phase3` が参照する最小限の args 名前空間を組み立てる。"""
    defaults = {
        "target": target,
        "output": str(output),
        "redownload": False,
        "open_browser": False,
        "emit_json": False,
        "quiet": True,
        "resolve_only": False,
        "asr": True,
        "asr_backend": "faster",
        "model": "turbo",
        "skip_asr": True,
        "no_hint": False,
        "refresh_members": False,
        "no_glossary": False,
        "glossary": None,
        "llm_context": False,
        "llm_correct": False,
        "llm_backend": "mlx",
        "llm_model": None,
        "llm_base_url": "http://localhost:8000/v1",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


_FAKE_META = {
    "deli_id": "99999",
    "date": "2026-05-15",
    "title": "テスト委員会",
    "speakers": [
        {"start": 0.0, "name": "山田太郎", "group": "委員長"},
        {"start": 100.0, "name": "テスト議員", "group": "テスト党"},
    ],
    "agenda": [],
    "page_url": "https://example.com/?deli_id=99999",
}


def test_skip_asr_missing_transcript_raises_invalid_input(monkeypatch, tmp_path):
    """transcript.json が存在しない状態で --skip-asr は InvalidInputError。"""
    monkeypatch.setattr(
        shugiin_main, "_resolve", lambda target: ("http://x/m.m3u8", _FAKE_META)
    )
    args = _make_args("99999", tmp_path)
    with pytest.raises(InvalidInputError, match="transcript.json"):
        shugiin_main._run_phase3(args)


def test_skip_asr_reads_existing_transcript_and_renders(monkeypatch, tmp_path):
    """transcript.json があれば Whisper 起動なしで HTML を生成。"""
    # 期待される base 名から transcript.json パスを逆算
    base = shugiin_main._safe_filename(
        f"{_FAKE_META['date']}_{_FAKE_META['title']}_衆{_FAKE_META['deli_id']}"
    )
    transcript_path = tmp_path / f"{base}_transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "deli_id": "99999",
                "title": "テスト委員会",
                "date": "2026-05-15",
                "hint": "テスト委員会の hint",
                "cues": [
                    {"start": 0.5, "end": 5.0, "text": "山田太郎委員長です。会議を開きます。"},
                    {"start": 105.0, "end": 110.0, "text": "テスト議員質問します。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        shugiin_main, "_resolve", lambda target: ("http://x/m.m3u8", _FAKE_META)
    )
    # transcribe や hls_to_wav が呼ばれたらテスト失敗 (skip 経路にならない証拠)
    from kokkai.shugiin import asr as asr_mod
    from kokkai.shugiin import audio as audio_mod

    def _fail(*a, **kw):
        raise AssertionError("--skip-asr 中に呼ばれてはならない")

    monkeypatch.setattr(asr_mod, "transcribe", _fail)
    monkeypatch.setattr(audio_mod, "hls_to_wav", _fail)
    # glossary / preclean のデフォルト辞書は普通に走らせて OK

    args = _make_args("99999", tmp_path, no_glossary=True)  # 辞書置換も切って純粋な経路に
    html_path, meta, stats, files = shugiin_main._run_phase3(args)

    # HTML が生成された
    assert html_path.exists()
    assert html_path.name.endswith("_asr.html")
    body = html_path.read_text(encoding="utf-8")
    # 復元した cue text が HTML 内に含まれる
    assert "会議を開きます" in body
    assert "質問します" in body

    # stats は skip 経路を示している
    assert stats["skip_asr"] is True
    assert stats["asr_backend"] is None
    assert stats["restored_from"].endswith("_transcript.json")


def test_full_asr_round_trips_transcript_json(monkeypatch, tmp_path):
    """通常 ASR 走行後に transcript.json が保存され、次回 --skip-asr で読める。"""
    monkeypatch.setattr(
        shugiin_main, "_resolve", lambda target: ("http://x/m.m3u8", _FAKE_META)
    )
    monkeypatch.setattr(
        shugiin_main, "_safe_filename",
        shugiin_main._safe_filename,  # 実装そのまま
    )
    # ffmpeg / hint / Whisper を全部 stub
    from kokkai.shugiin import asr as asr_mod
    from kokkai.shugiin import audio as audio_mod
    from kokkai.shugiin import hints as hints_mod
    from kokkai.shugiin import members as members_mod

    monkeypatch.setattr(audio_mod, "hls_to_wav", lambda *a, **kw: tmp_path / "fake.wav")
    monkeypatch.setattr(members_mod, "load_members", lambda refresh=False: [])
    monkeypatch.setattr(
        hints_mod, "build_initial_prompt",
        lambda meta, members: "テスト委員会の hint",
    )
    fake_cues = [
        {"start": 0.5, "end": 5.0, "text": "山田太郎委員長です。"},
        {"start": 105.0, "end": 110.0, "text": "質問します。"},
    ]
    monkeypatch.setattr(
        asr_mod, "transcribe", lambda *a, **kw: list(fake_cues)
    )

    # 1 回目: 通常 ASR 走行 → transcript.json が保存される
    args = _make_args("99999", tmp_path, skip_asr=False, no_glossary=True)
    html_path, _, stats, _files = shugiin_main._run_phase3(args)
    assert stats["skip_asr"] is False

    base = shugiin_main._safe_filename(
        f"{_FAKE_META['date']}_{_FAKE_META['title']}_衆{_FAKE_META['deli_id']}"
    )
    transcript_path = tmp_path / f"{base}_transcript.json"
    assert transcript_path.exists()
    saved = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert saved["deli_id"] == "99999"
    assert saved["hint"] == "テスト委員会の hint"
    assert len(saved["cues"]) == 2
    assert saved["cues"][0]["text"] == "山田太郎委員長です。"

    # 2 回目: --skip-asr で再 render (Whisper を呼ばないことを stub で保証)
    monkeypatch.setattr(
        asr_mod, "transcribe",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("呼ばれてはならない")),
    )
    args2 = _make_args("99999", tmp_path, skip_asr=True, no_glossary=True)
    html_path2, _, stats2, _files2 = shugiin_main._run_phase3(args2)
    assert stats2["skip_asr"] is True
    assert html_path2 == html_path
