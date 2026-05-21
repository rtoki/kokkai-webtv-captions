"""CLI: ``kkcap list`` — 両院の中継一覧を表示 (取込済 / 未取込を識別)。

ユーザーが sid/deli_id を知らなくても、「最近の中継」を発見して取り込める。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

from . import render as _render
from . import sangiin_list, shugiin_list, status as _status


def _parse_date(s: str) -> date:
    """YYYY-MM-DD or YYYY/MM/DD or YYYYMMDD or YYYY年M月D日 を date に。"""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # 日本語形式
    import re
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return date(y, mo, d)
    raise argparse.ArgumentTypeError(f"日付フォーマットを認識できません: {s}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kkcap-list",
        description=(
            "両院 (参議院 webtv / 衆議院 shugiintv) の中継一覧を取得。"
            " sid/deli_id を知らなくても、最近の中継を発見できる。"
        ),
    )
    parser.add_argument("--date", type=_parse_date,
                        help="特定日 (例: 2026-05-14)。指定しない場合は --since/--until で範囲指定 or 直近 7 日")
    parser.add_argument("--since", type=_parse_date,
                        help="この日付以降 (--date と併用不可)")
    parser.add_argument("--until", type=_parse_date,
                        help="この日付以前")
    parser.add_argument("--house", choices=["sangiin", "shugiin"],
                        help="院でフィルタ")
    parser.add_argument("--committee", default=None,
                        help="委員会名の部分一致フィルタ (例: 法務)")
    parser.add_argument("--only-new", action="store_true",
                        help="未取込のものだけ表示")
    parser.add_argument("--only-fetched", action="store_true",
                        help="取込済のものだけ表示")
    parser.add_argument("-o", "--output", default="out",
                        help="取込済判定で参照するディレクトリ (既定: ./out)")
    g_json = parser.add_mutually_exclusive_group()
    g_json.add_argument("--json", dest="emit_json", action="store_true",
                        help="1 行 JSON で結果を stdout に")
    g_json.add_argument("--jsonl", dest="emit_jsonl", action="store_true",
                        help="サマリ 1 行 + 項目 1 行ずつの JSONL")
    return parser


def _resolve_range(args: argparse.Namespace) -> tuple[date, date]:
    today = date.today()
    if args.date:
        return args.date, args.date
    if args.since or args.until:
        s = args.since or today - timedelta(days=7)
        u = args.until or today
        return s, u
    # 既定: 直近 7 日
    return today - timedelta(days=7), today


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    start, end = _resolve_range(args)
    items: list[dict] = []
    if args.house in (None, "shugiin"):
        items.extend(shugiin_list.fetch_for_range(start, end))
    if args.house in (None, "sangiin"):
        items.extend(sangiin_list.fetch_for_range(start, end))

    # 取込済判定
    items = _status.annotate(items, Path(args.output))

    # フィルタ
    if args.committee:
        items = [it for it in items if args.committee in it.get("title", "")]
    if args.only_new:
        items = [it for it in items if not it.get("fetched")]
    if args.only_fetched:
        items = [it for it in items if it.get("fetched")]

    if args.emit_jsonl:
        print(_render.render_jsonl(items))
    elif args.emit_json:
        print(_render.render_json(items))
    else:
        print(_render.render_human(items))


if __name__ == "__main__":
    main()
