"""errors.py の exit code マッピングと CliError 階層のテスト。"""

from __future__ import annotations

from kokkai.errors import (
    AsrError,
    CliError,
    FetchError,
    InvalidInputError,
    LlmError,
    MissingToolError,
)


def test_exit_code_mapping_is_stable():
    """CLI / agent と契約している exit code を固定する。"""
    assert CliError.code == 1
    assert InvalidInputError.code == 2
    assert FetchError.code == 3
    assert MissingToolError.code == 4
    assert AsrError.code == 5
    assert LlmError.code == 6


def test_all_inherit_from_cli_error():
    for cls in (InvalidInputError, FetchError, MissingToolError, AsrError, LlmError):
        assert issubclass(cls, CliError)
        assert issubclass(cls, Exception)


def test_errors_carry_message():
    err = InvalidInputError("sid が不正です: abc")
    assert "sid が不正" in str(err)
    assert err.code == 2
