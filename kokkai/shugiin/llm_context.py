"""
Whisper 2-pass のための「pass1 出力から固有名詞を抽出 → pass2 prompt に注入」を担う。

参考: 「Whisper: Courtside Edition — Enhancing ASR Performance Through LLM-Driven
Context Generation」 (arXiv 2602.18966) の縮退版。

使い方:
    from .llm_context import extract_glossary_terms, merge_into_prompt

    pass1 = transcribe(wav, initial_prompt=base_prompt)
    terms = extract_glossary_terms(
        " ".join(c["text"] for c in pass1),
        backend="mlx",
    )
    augmented = merge_into_prompt(base_prompt, terms)
    pass2 = transcribe(wav, initial_prompt=augmented)

Whisper の ``initial_prompt`` は 224 token 上限なので、抽出語数は ``MAX_TERMS=10``
に絞る (実測で限界を超えると pass2 が degenerate になりやすい)。
"""

from __future__ import annotations

import json
import re
import sys


# llm_correct と backend / model を共有
from .llm_correct import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_MLX_MODEL,
    _get_mlx_model,
    default_model_for,
)


MAX_INPUT_CHARS = 80_000
MAX_TERMS = 10


SYSTEM_PROMPT = """あなたは日本語音声書き起こしの校正アシスタントです。
与えられた transcript (Whisper による第 1 パス出力) から、ASR が誤認しがちな語を
抽出してください:
- 固有名詞 (人名・組織名・地名・略号)
- 専門用語・業界用語・法令略称
- カタカナ語・外来語
- 機関名や役職名

出力は ``{"terms": [...]}`` の JSON オブジェクト。重複は除き、重要度が高い順に
最大 10 個まで。例:
{"terms": ["外為法", "もんじゅ", "AISI", "国家サイバーセキュリティ戦略本部", "対内直接投資"]}"""


def _build_terms_schema():
    from pydantic import BaseModel, Field
    from typing import Annotated

    class TermsResponse(BaseModel):
        terms: Annotated[list[str], Field(max_length=MAX_TERMS)] = []

    return TermsResponse


# ============================================================================
# Backend: mlx + Outlines
# ============================================================================


def _extract_with_mlx(text: str, *, model_id: str) -> list[str]:
    from ..errors import LlmError

    schema = _build_terms_schema()
    model, tokenizer = _get_mlx_model(model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text[:MAX_INPUT_CHARS]},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception as e:  # noqa: BLE001
        raise LlmError(
            f"tokenizer.apply_chat_template 失敗 (instruct モデルですか?): {e}"
        ) from e
    try:
        raw = model(prompt, output_type=schema, max_tokens=512)
    except Exception as e:  # noqa: BLE001
        raise LlmError(f"Outlines 推論失敗: {e}") from e
    return _parse_terms(raw)


# ============================================================================
# 共通
# ============================================================================


def _parse_terms(raw: str) -> list[str]:
    raw = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(
            f"[llm-context] WARN: JSON parse 失敗、抽出をスキップ (raw: {raw[:200]!r})",
            file=sys.stderr,
        )
        return []
    terms = data.get("terms") if isinstance(data, dict) else data
    if not isinstance(terms, list):
        return []
    return [t for t in terms if isinstance(t, str) and t][:MAX_TERMS]


def extract_glossary_terms(
    text: str,
    *,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
) -> list[str]:
    """transcript text から固有名詞・専門語を最大 ``MAX_TERMS`` 個抽出。"""
    from ..errors import LlmError

    if not text:
        return []
    if backend not in BACKENDS:
        raise LlmError(f"未対応の backend: {backend}")

    model_id = model or default_model_for(backend)
    print(
        f"[llm-context] 固有名詞抽出中 (backend={backend}, model={model_id})",
        file=sys.stderr,
    )
    if backend == "mlx":
        return _extract_with_mlx(text, model_id=model_id)
    # openai backend は未サポート (構造保証が弱く 2-pass の品質に直結するため)
    raise LlmError(
        "llm-context は backend=mlx のみ対応 "
        "(openai 互換 HTTP は構造保証が弱く 2-pass 用途には不向き)"
    )


def merge_into_prompt(base_prompt: str | None, terms: list[str]) -> str:
    """既存 ``initial_prompt`` に抽出語を追加。重複は除外、長さは concat で 224
    token (約 600 char) を超えないよう適度に切る。"""
    if not terms:
        return base_prompt or ""
    existing = base_prompt or ""
    new = [t for t in terms if t not in existing]
    if not new:
        return existing
    joined = "、".join(new)
    # 224 token ≈ 600 char (日本語 1 文字 ≈ 2-3 token 弱) を上限の目安に
    candidate = (existing + " " if existing else "") + joined
    if len(candidate) > 600:
        candidate = candidate[:600]
    return candidate
