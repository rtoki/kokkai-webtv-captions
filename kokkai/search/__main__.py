"""CLI: ``kkcap search <query>`` (or ``python -m kokkai.search``).

DB を使わず、毎クエリで ``out/*.meta.json`` をスキャンして BM25 でランキング。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cache as _cache
from .index import iter_records
from .query import filter_records, score_records
from .render import render_human, render_json, render_jsonl
from .tokenize import tokenize_query


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kkcap-search",
        description=(
            "out/ に溜まった会議文字起こしを BM25 で全文検索。"
            " DB 不要 (毎クエリで out/*.meta.json をスキャン)。"
        ),
    )
    parser.add_argument(
        "query",
        help='検索クエリ。スペース区切り = AND、" " で囲むとフレーズ完全一致',
    )
    parser.add_argument(
        "-o", "--output", default="out",
        help="検索対象ディレクトリ (sangiin/shugiin の出力先、既定: ./out)",
    )
    parser.add_argument(
        "--topk", type=int, default=10,
        help="表示する上位件数 (既定: 10)",
    )

    g_filter = parser.add_argument_group("絞り込み")
    g_filter.add_argument("--since", help="この日付以降 (例: 2026-05-01)")
    g_filter.add_argument("--until", help="この日付以前 (例: 2026-05-31)")
    g_filter.add_argument("--speaker", help="発言者名の部分一致")
    g_filter.add_argument("--committee", help="会議名の部分一致 (例: 法務)")
    g_filter.add_argument(
        "--house", choices=["sangiin", "shugiin"],
        help="参議院 / 衆議院 で絞り込み",
    )

    g_cache = parser.add_argument_group("キャッシュ")
    g_cache.add_argument("--no-cache", action="store_true",
                         help="トークン化キャッシュを使わず毎回再計算")
    g_cache.add_argument("--rebuild-cache", action="store_true",
                         help="既存キャッシュを削除して再構築")

    g_suggest = parser.add_argument_group("未取込候補の提案")
    g_suggest.add_argument("--suggest", action="store_true",
                           help="検索結果の末尾に、議題名にクエリを含む未取込会議を提示")
    g_suggest.add_argument("--suggest-days", type=int, default=14,
                           help="未取込候補を探す直近日数 (既定: 14)")

    g_out = parser.add_argument_group("出力")
    g_json = g_out.add_mutually_exclusive_group()
    g_json.add_argument("--json", dest="emit_json", action="store_true",
                        help="1 行 JSON で結果を stdout に")
    g_json.add_argument("--jsonl", dest="emit_jsonl", action="store_true",
                        help="meta 1 行 + ヒット 1 行ずつの JSONL")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    if args.rebuild_cache:
        if _cache.clear_cache(out_dir):
            print(f"[search] キャッシュを削除しました: {_cache.cache_path(out_dir)}", file=sys.stderr)

    records = iter_records(out_dir, use_cache=not args.no_cache)
    if not records:
        msg = (
            f"out/*.meta.json が見つかりません: {out_dir}\n"
            f"先に sangiin / shugiin を 1 回以上実行して文字起こしを生成してください。"
        )
        if args.emit_json or args.emit_jsonl:
            import json
            print(json.dumps({"ok": False, "error": "NoIndex", "message": msg}, ensure_ascii=False))
        else:
            print(msg, file=sys.stderr)
        sys.exit(2)

    records = filter_records(
        records,
        since=args.since,
        until=args.until,
        speaker=args.speaker,
        committee=args.committee,
        house=args.house,
    )

    terms, phrases = tokenize_query(args.query)
    if not terms and not phrases:
        print("クエリから検索語を抽出できませんでした。", file=sys.stderr)
        sys.exit(2)

    hits = score_records(records, terms, phrases)[: args.topk]

    if args.emit_jsonl:
        print(render_jsonl(args.query, hits))
    elif args.emit_json:
        print(render_json(args.query, hits))
    else:
        print(render_human(args.query, hits))

    if args.suggest and not (args.emit_json or args.emit_jsonl):
        _print_suggestions(args.query, out_dir, days=args.suggest_days)


def _print_suggestions(query: str, out_dir: Path, *, days: int) -> None:
    """直近 days 日の未取込会議のうち、議題名に query を含むものを提示。"""
    from datetime import date, timedelta
    from ..list import sangiin_list, shugiin_list, status as list_status

    end = date.today()
    start = end - timedelta(days=days)
    items: list[dict] = []
    items.extend(shugiin_list.fetch_for_range(start, end))
    items.extend(sangiin_list.fetch_for_range(start, end))
    items = list_status.annotate(items, out_dir)
    # 未取込のみ + クエリの最初の語が会議名に含まれる
    first_term = (query.replace('"', "").split() or [""])[0]
    suggestions = [
        it for it in items
        if not it.get("fetched")
        and (not first_term or first_term in it.get("title", ""))
    ]
    if not suggestions:
        return

    print("\n--- 未取込候補 (直近 {} 日、議題名に '{}' を含む) ---".format(days, first_term))
    for it in suggestions[:10]:
        house = "参議院" if it.get("house") == "sangiin" else "衆議院"
        dur = f" / {it['duration']}" if it.get("duration") else ""
        print(f"  {it.get('date')} [{house}] {it.get('title')}{dur}")
        cmd = "sangiin" if it.get("house") == "sangiin" else "shugiin"
        print(f"    取込: kkcap fetch {it.get('id')}  (or `kkcap {cmd} {it.get('id')}`)")


if __name__ == "__main__":
    main()
