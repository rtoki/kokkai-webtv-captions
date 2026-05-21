"""``kokkai.shugiin._models`` (HF cache 確認 + Y/N プロンプト) のテスト。"""

from __future__ import annotations

import io

import pytest

from kokkai.errors import MissingToolError
from kokkai.shugiin import _models


def test_hf_model_cache_path_under_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(_models, "Path", _models.Path)  # smoke
    p = _models.hf_model_cache_path("mlx-community/whisper-large-v3-turbo")
    assert p.name == "models--mlx-community--whisper-large-v3-turbo"


def test_hf_cache_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "myhub"))
    assert _models.hf_cache_dir() == tmp_path / "myhub"

    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert _models.hf_cache_dir() == tmp_path / "hfhome" / "hub"


def test_is_model_cached_true_when_file_present(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--org--name" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_text("x")
    assert _models.is_model_cached("org/name")


def test_is_model_cached_false_when_only_empty_dirs(monkeypatch, tmp_path):
    """snapshots/ にディレクトリだけあって実ファイル無いケースは未 DL 扱い。"""
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    (tmp_path / "models--org--name" / "snapshots" / "rev1").mkdir(parents=True)
    assert not _models.is_model_cached("org/name")


def test_is_model_cached_false_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    assert not _models.is_model_cached("org/name")


def test_ensure_model_downloaded_skips_when_cached(monkeypatch, tmp_path):
    """既に cached なら prompt せず即 return (input は呼ばれない)。"""
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--org--name" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_text("x")

    def _no_input():
        raise AssertionError("input() should not be called for cached models")

    monkeypatch.setattr("builtins.input", _no_input)
    _models.ensure_model_downloaded("org/name", label="asr")


def test_ensure_model_downloaded_skips_when_non_interactive(monkeypatch, tmp_path):
    """非対話 (TTY ではない) なら未 cached でも素通り。"""
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(_models, "_is_interactive", lambda: False)

    def _no_input():
        raise AssertionError("input() should not be called when non-interactive")

    monkeypatch.setattr("builtins.input", _no_input)
    _models.ensure_model_downloaded("org/name", label="asr")


def test_ensure_model_downloaded_yes_proceeds(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(_models, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "y")
    _models.ensure_model_downloaded("org/name", label="asr", size_hint="1.5 GB")
    err = capsys.readouterr().err
    assert "org/name" in err
    assert "1.5 GB" in err


def test_ensure_model_downloaded_empty_input_proceeds(monkeypatch, tmp_path):
    """空 Enter (= デフォルト Y) で続行できる。"""
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(_models, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "")
    _models.ensure_model_downloaded("org/name", label="asr")


def test_ensure_model_downloaded_no_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(_models, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "n")
    with pytest.raises(MissingToolError):
        _models.ensure_model_downloaded("org/name", label="asr")


def test_ensure_model_downloaded_eof_proceeds(monkeypatch, tmp_path):
    """input() が EOFError (パイプ EOF など) でも素直に続行する。"""
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(_models, "_is_interactive", lambda: True)

    def _eof():
        raise EOFError()

    monkeypatch.setattr("builtins.input", _eof)
    _models.ensure_model_downloaded("org/name", label="asr")
