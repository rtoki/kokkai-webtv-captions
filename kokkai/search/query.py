"""
BM25 スコアリング (自前実装、追加依存なし)。

参考: Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework: BM25 and Beyond"
標準パラメータ k1=1.5, b=0.75。

API:
- ``score_records(records, terms, phrases, k1=1.5, b=0.75) -> list[(rec, score)]``
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .tokenize import tokenize


def score_records(
    records: list[dict],
    terms: list[str],
    phrases: list[str] | None = None,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[dict, float]]:
    """
    各レコード (cue) を BM25 スコアリングし、(record, score) 降順で返す。

    Args:
        records: ``index.iter_records()`` の出力
        terms: 検索クエリの内容語リスト (AND マッチ前提だが、BM25 は OR の重み付け)
        phrases: 必須含有フレーズ (substring 一致でフィルタ)

    Returns:
        スコア > 0 のレコードのみ降順ソート
    """
    if not records or not terms:
        return []

    phrases = phrases or []

    # 各 record の token list を取得 (キャッシュ済の "tokens" があれば再 tokenize しない)
    docs: list[list[str]] = [
        r["tokens"] if r.get("tokens") else tokenize(r["text"])
        for r in records
    ]
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
    N = len(docs)

    # 各 term の document frequency
    df: Counter[str] = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1

    # 各 term の IDF
    idf = {
        term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
        for term in set(terms) if df[term] > 0
    }

    out: list[tuple[dict, float]] = []
    for rec, d, dl in zip(records, docs, doc_lens):
        # phrase が指定されていたら全部含むレコードだけ通す
        text = rec["text"]
        if any(p not in text for p in phrases):
            continue

        # term frequency
        tf = Counter(d)
        score = 0.0
        for term in terms:
            if term not in idf:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            norm = 1 - b + b * (dl / avgdl if avgdl > 0 else 0)
            score += idf[term] * (f * (k1 + 1)) / (f + k1 * norm)

        if score > 0:
            out.append((rec, score))

    out.sort(key=lambda x: -x[1])
    return out


def filter_records(
    records: list[dict],
    *,
    since: str | None = None,
    until: str | None = None,
    speaker: str | None = None,
    committee: str | None = None,
    house: str | None = None,
) -> list[dict]:
    """日付範囲・発言者・委員会・院でレコードを絞り込む。

    日付は ``2026-05-15`` / ``2026年5月15日`` どちらでも受ける (来た文字列をパース、
    途中の柔軟性を確保)。
    """
    def _norm_date(s: str) -> str:
        # 「2026年5月15日」 → 「2026-05-15」
        import re
        m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
        if m:
            y, mo, d = m.groups()
            return f"{y}-{int(mo):02d}-{int(d):02d}"
        return s.strip()

    def _date_of(rec: dict) -> str:
        return _norm_date(rec.get("date", ""))

    out = records
    if since:
        s = _norm_date(since)
        out = [r for r in out if _date_of(r) >= s]
    if until:
        u = _norm_date(until)
        out = [r for r in out if _date_of(r) <= u]
    if speaker:
        out = [r for r in out if speaker in (r.get("speaker_name") or "")]
    if committee:
        out = [r for r in out if committee in (r.get("title") or "")]
    if house:
        out = [r for r in out if r.get("house") == house]
    return out
