"""
ASR 出力を LLM で事後校正するモジュール (Phase 4)。

3 段構成:

1. ``preclean_loops``: Whisper の degenerate loop (``ティティティ`` 等) を
   ``[音声不明瞭]`` に置換。文字種ユニーク率が低い長文を heuristic で検出。
2. 静的 glossary (``glossary.py``) で確実な誤訳を deterministic に置換
   (この呼び出しは ``__main__.py`` から行う)。
3. ``correct_cues``: ローカル LLM で context-aware に校正。

LLM backend (``--llm-backend``):

- **``mlx``** (推奨, ローカル, Apple Silicon): ``mlx-lm`` + `Outlines
  <https://github.com/dottxt-ai/outlines>`_ で **token レベル grammar-constrained
  generation**。Pydantic schema を満たさない token を生成不能化することで、9B
  クラスのモデルでも JSON 構造違反が起こらない。
- **``openai``** (OpenAI 互換 HTTP): vllm-mlx / Ollama / LM Studio で
  ``response_format={"type": "json_object"}`` を使う。後方互換用に残してある
  だけで、構造保証は緩い (echo モード等で時々壊れる)。

参考:
- arXiv 2409.06062 (RAG-based 固有名詞修正)
- arXiv 2408.16180 (Japanese MPA GER)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request


BACKENDS = ("mlx", "openai")
DEFAULT_BACKEND = "mlx"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_TIMEOUT = 120
# 1 リクエストに詰める cue 数。多いと LLM 入出力が長くなり遅くなる/失敗しやすい。
# 議会発言は 1 cue で 100-500 文字なので 8 cue ≈ 1000-4000 char の入力。
DEFAULT_BATCH_SIZE = 8


def default_model_for(backend: str) -> str:
    return {
        "mlx": DEFAULT_MLX_MODEL,
        "openai": DEFAULT_MLX_MODEL,
    }.get(backend, DEFAULT_MLX_MODEL)


SYSTEM_PROMPT = """あなたは日本語の国会音声書き起こしの校正アシスタントです。
Whisper が生成した cue を受け取り、文脈から明らかな誤認識のみを修正します。

修正してよいもの:
- 文脈から明らかに意味の通らない固有名詞・専門用語の誤り
  (法律用語・議事手続き語などで、議会文脈なら一意に正解が決まるもの)
- 漢字選択ミス (音は合っているが意味が違う) で、文脈から正解が明らか
- 同音異義語の文脈不整合
- glossary に挙げた語に音声的に近い誤りで確証が高いもの

修正してはいけないもの:
- 文体・敬語・口語表現の整形
- 句読点の追加・削除 (既に挿入済)
- 文の分割・併合 (cue 構造は維持)
- 自信のない箇所 (元のまま残す)
- 話者の言い間違いや言いよどみ (それは正しい書き起こし)
- ``[音声不明瞭]`` プレースホルダ (そのまま残す)

出力は JSON オブジェクト ``{"corrections": [{"id": int, "text": str}, ...]}``。
修正が不要な cue は配列に含めない。"""


# ============================================================================
# Whisper degenerate loop の preclean
# ============================================================================


def _is_repetitive_loop(text: str) -> bool:
    """Whisper の degenerate loop を heuristic で検出。

    200 char 超 かつ ユニーク文字 / 全文字 < 5% なら確実にループ。
    例: 「ティティティ...」(unique 1)、「はいはい...」(unique 2)。
    通常の日本語文は unique 率 30% 以上。
    """
    if len(text) <= 200:
        return False
    return len(set(text)) / len(text) < 0.05


# 通常文の途中に 1-4 文字の音節が 10 回以上連続反復するパターン
# (例: 「AIセーフティ・インシティティティティティ...他社との」)。
# 全体のユニーク文字率で見ると正常範囲に入ってしまうため、別パスで検出する。
_INLINE_LOOP_RE = re.compile(r"(.{1,4}?)\1{9,}")


def _strip_inline_loops(text: str) -> tuple[str, int]:
    """text 中のインライン degenerate loop を ``[音声不明瞭]`` 1 つに圧縮。

    Returns: ``(cleaned_text, n_replacements)``
    """
    n = 0

    def _repl(_m: "re.Match[str]") -> str:
        nonlocal n
        n += 1
        return "[音声不明瞭]"

    cleaned = _INLINE_LOOP_RE.sub(_repl, text)
    return cleaned, n


def preclean_loops(cues: list[dict]) -> int:
    """Whisper degenerate loop を ``[音声不明瞭]`` に置換 (破壊的)。

    2 段で検出する:

    1. cue 全体が単一文字種のループ (`_is_repetitive_loop`) → cue 全体置換
    2. cue 内で 1-4 文字音節が 10 回以上連続反復 (`_strip_inline_loops`) →
       該当部分のみ ``[音声不明瞭]`` 1 つに置換 (前後の正常文は保持)

    元 text は最初の置換時に ``_original_text`` に退避する。修正件数 (置換が
    1 つ以上起きた cue の数) を返す。
    """
    count = 0
    for c in cues:
        text = c.get("text", "")
        if _is_repetitive_loop(text):
            c["_original_text"] = text
            c["text"] = "[音声不明瞭]"
            count += 1
            continue
        cleaned, n = _strip_inline_loops(text)
        if n:
            c["_original_text"] = text
            c["text"] = cleaned
            count += 1
    return count


# ============================================================================
# YouTube 系幻覚句の除外 (Whisper が無音・開会前 BGM で吐き出す決まり文句)
# ============================================================================

# Whisper は YouTube 動画の終了テロップで大量学習しているため、無音/低 SNR/
# 開会前 BGM などの区間で以下のフレーズを高頻度で hallucinate する。
# 文脈とは無関係に出るので、cue 単位で削る。
_HALLUCINATION_PHRASES = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご清聴ありがとうございました",
    "視聴ありがとうございました",
    "字幕視聴ありがとうございました",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "高評価よろしくお願いします",
    "Thank you for watching",
    "Thanks for watching",
)

_TRIM_PUNCT = "。、 \t\n　"

# 短いフレーズ (2-40 文字) が連続 3 回以上現れて text 全体を覆うパターン
# (mlx-whisper が 1 segment に幻覚句を多数詰める場合の検出)
_REPEAT_PHRASE_RE = re.compile(r"(.{2,40}?)\1{2,}")


def _is_hallucination_text(text: str) -> bool:
    t = text.strip().strip(_TRIM_PUNCT).strip()
    if not t:
        return False
    if t in _HALLUCINATION_PHRASES:
        return True
    # 同一フレーズの連結 (テキスト全体が短いフレーズの繰り返しだけ)
    m = _REPEAT_PHRASE_RE.fullmatch(t)
    if m:
        return True
    # 幻覚句が含まれ、それ以外がほぼ無いケース
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in t:
            residual = t.replace(phrase, "").strip(_TRIM_PUNCT).strip()
            if not residual:
                return True
    return False


def drop_hallucinations(cues: list[dict]) -> int:
    """Whisper の YouTube 系幻覚 cue を破壊的に除去する。

    Returns: 除去した cue 数。
    """
    keep: list[dict] = []
    dropped = 0
    for c in cues:
        if _is_hallucination_text(c.get("text", "")):
            dropped += 1
            continue
        keep.append(c)
    if dropped:
        cues[:] = keep
    return dropped


# ============================================================================
# 共通: LLM 入力 / 応答パース
# ============================================================================


def _build_user_message(cues_batch: list[dict], context_hint: str | None) -> str:
    """LLM への入力 (会議メタ + cue 群) を組み立てる。"""
    payload: dict = {
        "cues": [
            {"id": c.get("id", i), "text": c["text"]}
            for i, c in enumerate(cues_batch)
        ],
    }
    if context_hint:
        payload["context"] = context_hint
    return json.dumps(payload, ensure_ascii=False)


def _parse_corrections(raw: str) -> list[dict]:
    """LLM の JSON 応答から ``[{"id": int, "text": str}, ...]`` を取り出す。"""
    raw = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(
            f"[llm-correct] WARN: JSON parse 失敗、この batch を skip: {e}"
            f" (raw: {raw[:200]!r})",
            file=sys.stderr,
        )
        return []
    corrections = data.get("corrections") if isinstance(data, dict) else data
    if not isinstance(corrections, list):
        return []
    return [
        d for d in corrections
        if isinstance(d, dict) and "id" in d and "text" in d
    ]


# ============================================================================
# Backend: mlx + Outlines (grammar-constrained, 推奨)
# ============================================================================


def _load_mlx_outlines():
    """Outlines + mlx-lm を遅延 import (`pip install 'outlines[mlxlm]'`)。"""
    from ..errors import MissingToolError
    try:
        import mlx_lm  # type: ignore[import-not-found]
        import outlines  # type: ignore[import-not-found]
    except ImportError as e:
        raise MissingToolError(
            "Outlines (mlx-lm 拡張) が必要です。\n"
            "  pip install 'outlines[mlxlm]'"
        ) from e
    return mlx_lm, outlines


def _build_correction_schema():
    """Pydantic schema (Outlines に渡す grammar)。"""
    from pydantic import BaseModel

    class Correction(BaseModel):
        id: int
        text: str

    class CorrectionResponse(BaseModel):
        corrections: list[Correction]

    return CorrectionResponse


_MLX_STATE: dict = {"model_id": None, "model": None, "tokenizer": None}


_KNOWN_LLM_MODEL_SIZES = {
    "mlx-community/Qwen3.5-9B-MLX-4bit": "5.5 GB",
}


def _get_mlx_model(model_id: str):
    """mlx-lm モデルをプロセス単位でキャッシュ (batch 毎の reload を避ける)。"""
    from ..errors import LlmError
    from ._models import ensure_model_downloaded

    if _MLX_STATE.get("model_id") == model_id and _MLX_STATE.get("model") is not None:
        return _MLX_STATE["model"], _MLX_STATE["tokenizer"]
    # mlx-lm の DL は load() 内部で走るので、その前に Y/N (対話 TTY のみ)。
    ensure_model_downloaded(
        model_id,
        label="llm-correct",
        size_hint=_KNOWN_LLM_MODEL_SIZES.get(model_id),
    )
    mlx_lm, outlines = _load_mlx_outlines()
    print(
        f"[llm-correct] mlx+outlines モデルロード中: {model_id}",
        file=sys.stderr,
    )
    try:
        raw_model, tokenizer = mlx_lm.load(model_id)
        model = outlines.from_mlxlm(raw_model, tokenizer)
    except Exception as e:  # noqa: BLE001
        raise LlmError(f"mlx-lm / Outlines のモデルロード失敗 ({model_id}): {e}") from e
    _MLX_STATE.update({"model_id": model_id, "model": model, "tokenizer": tokenizer})
    return model, tokenizer


def _correct_with_mlx(
    cues_batch: list[dict],
    context_hint: str | None,
    *,
    model_id: str,
) -> str:
    """Outlines + mlx-lm で grammar-constrained に JSON を生成。"""
    from ..errors import LlmError

    schema = _build_correction_schema()
    model, tokenizer = _get_mlx_model(model_id)
    user_content = _build_user_message(cues_batch, context_hint)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
        return model(prompt, output_type=schema, max_tokens=8192)
    except Exception as e:  # noqa: BLE001
        raise LlmError(f"Outlines 推論失敗: {e}") from e


# ============================================================================
# Backend: openai 互換 HTTP (vllm-mlx / Ollama / LM Studio)
# ============================================================================


def _correct_with_openai_http(
    cues_batch: list[dict],
    context_hint: str | None,
    *,
    base_url: str,
    model: str,
    timeout: int,
    api_key: str | None,
) -> str:
    from ..errors import LlmError
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(cues_batch, context_hint)},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise LlmError(
            f"LLM サーバへの接続失敗 ({url}): {e}\n"
            f"vllm-mlx が起動しているか確認してください:\n"
            f"  pip install vllm-mlx\n"
            f"  vllm-mlx serve {model} --port 8000"
        ) from e

    resp = json.loads(raw)
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LlmError(f"LLM の応答形式が想定外: {raw[:500]}") from e


# ============================================================================
# 公開 API
# ============================================================================


def correct_cues(
    cues: list[dict],
    *,
    context_hint: str | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    base_url: str = DEFAULT_LLM_BASE_URL,
    api_key: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: int = DEFAULT_LLM_TIMEOUT,
) -> int:
    """
    cue リストを LLM で校正 (破壊的)。

    Args:
        cues: ``[{"id"?, "start", "end", "text"}, ...]``。id 無ければ index 自動付与。
        context_hint: 会議メタ等の追加コンテキスト
        backend: ``mlx`` / ``openai``
        model: モデル名。省略時は backend ごとの既定 (``default_model_for``)
        base_url: openai backend のみで使用 (OpenAI 互換 endpoint)
        api_key: openai backend で使用
        batch_size: 1 リクエストの cue 数
        timeout: openai backend の HTTP timeout (秒)

    Returns:
        修正された cue の件数
    """
    from ..errors import LlmError

    if not cues:
        return 0
    if backend not in BACKENDS:
        raise LlmError(f"未対応の backend: {backend} (使えるのは {BACKENDS})")

    for i, c in enumerate(cues):
        c.setdefault("id", i)
    by_id = {c["id"]: c for c in cues}

    model_id = model or default_model_for(backend)
    n_batches = (len(cues) + batch_size - 1) // batch_size
    print(
        f"[llm-correct] {len(cues)} cue を {n_batches} batch で校正中"
        f" (backend={backend}, model={model_id})",
        file=sys.stderr,
    )

    total_changed = 0
    for batch_i in range(n_batches):
        batch = cues[batch_i * batch_size : (batch_i + 1) * batch_size]
        if backend == "mlx":
            raw = _correct_with_mlx(batch, context_hint, model_id=model_id)
        else:  # openai
            raw = _correct_with_openai_http(
                batch, context_hint,
                base_url=base_url, model=model_id,
                timeout=timeout, api_key=api_key,
            )
        corrections = _parse_corrections(raw)
        for c in corrections:
            sid = c["id"]
            new_text = str(c["text"]).strip()
            if sid in by_id and new_text and new_text != by_id[sid]["text"]:
                by_id[sid]["text"] = new_text
                total_changed += 1
        print(
            f"[llm-correct]   batch {batch_i + 1}/{n_batches}: {len(corrections)} 件修正",
            file=sys.stderr,
        )

    print(f"[llm-correct] 完了: {total_changed}/{len(cues)} cue を修正", file=sys.stderr)
    return total_changed
