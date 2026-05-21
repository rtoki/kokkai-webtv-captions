"""CLI が exit code に変換する例外群 (sangiin / shugiin 共通)。

| code | 意味                              | 例                                                |
| ---- | --------------------------------- | ------------------------------------------------- |
| 0    | 成功                              |                                                   |
| 1    | その他/未分類                     | 予期しない例外                                    |
| 2    | 入力不正                          | sid / deli_id が不正、ページ構造変化              |
| 3    | ネットワーク/取得失敗             | HTTP error、HLS/VTT 取得失敗                      |
| 4    | 外部ツール欠如                    | ffmpeg / faster-whisper / mlx_whisper / outlines  |
| 5    | ASR 失敗                          | Whisper 推論失敗                                  |
| 6    | LLM 校正/補強失敗 (non-fatal)     | --llm-context / --llm-correct の推論失敗          |
| 130  | キーボード中断                    | Ctrl-C                                            |
"""

from __future__ import annotations


class CliError(Exception):
    code: int = 1


class InvalidInputError(CliError):
    code = 2


class FetchError(CliError):
    code = 3


class MissingToolError(CliError):
    code = 4


class AsrError(CliError):
    code = 5


class LlmError(CliError):
    code = 6
