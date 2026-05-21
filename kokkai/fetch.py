"""``kkcap fetch <id> ...`` — 院を自動判定して sangiin / shugiin に dispatch する薄ラッパ。

id の形式で院を判定:
- 参議院 sid: 4-5 桁 (1000-9999、稀に 10000+)。webtv.sangiin.go.jp の sid 体系
- 衆議院 deli_id: 5 桁、概ね 50000+ (shugiintv の deli_id 体系)

混在指定可能: ``kkcap fetch 8955 56239``

URL を直接渡しても OK (sid=N / deli_id=N から判定)。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta


def _parse_date(s: str) -> date:
    """YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD / YYYY年M月D日 を date に (kkcap-list と同形式)."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return date(y, mo, d)
    raise argparse.ArgumentTypeError(f"日付フォーマットを認識できません: {s}")


def _classify(target: str) -> str:
    """target が sangiin (sid) か shugiin (deli_id) かを判定。判定不能なら ValueError。"""
    s = target.strip()
    # URL から
    if "sangiin.go.jp" in s or "sid=" in s:
        return "sangiin"
    if "shugiintv.go.jp" in s or "deli_id=" in s:
        return "shugiin"
    # 数値 ID から
    if s.isdigit():
        n = int(s)
        # 経験則: shugiin deli_id は概ね 30000 以上、sangiin sid は 1-19999
        # 境界は将来動くかもしれないが、現状の運用では明確に分かれる
        if n >= 30000:
            return "shugiin"
        return "sangiin"
    raise ValueError(f"sid/deli_id/URL のいずれにも見えません: {target!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kkcap-fetch",
        description=(
            "複数の sid (参議院) / deli_id (衆議院) を一括取込。"
            " 院は ID 形式から自動判定。"
        ),
    )
    parser.add_argument("target", nargs="*", default=[],
                        help="sid / deli_id / URL を 0 個以上 (例: 8955 56239)。"
                             " --from/--to 併用時は日付列挙にマージ")
    parser.add_argument("--from", dest="date_from", type=_parse_date, default=None,
                        help="この日付以降の中継を列挙して fetch (例: 2026-04-20)。"
                             " --to 省略時は今日まで")
    parser.add_argument("--to", dest="date_to", type=_parse_date, default=None,
                        help="この日付以前の中継を列挙して fetch。"
                             " --from 省略時は --to の 7 日前から")
    parser.add_argument("--house", choices=["sangiin", "shugiin"], default=None,
                        help="日付列挙を片院に絞る")
    parser.add_argument("--asr", dest="asr", action="store_true", default=True,
                        help="衆議院は ASR を実行 (既定で有効、要 [shugiin-asr] + ffmpeg)")
    parser.add_argument("--no-asr", dest="asr", action="store_false",
                        help="衆議院も ASR を実行せず Phase 2 のタイムラインのみ生成")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="既に取込済 (meta.json + html) ならスキップ")
    parser.add_argument("-o", "--output", default="out",
                        help="出力ディレクトリ (既定: ./out)")
    parser.add_argument("--no-open", action="store_true",
                        help="生成後にブラウザを開かない")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch せず、対象 ID を <院>\\t<id>[\\t<日付 タイトル>] で表示")
    return parser


def _enumerate_by_date(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """``--from/--to`` で指定された期間の中継 ID を列挙。

    Returns: ``[(house, id, "YYYY-MM-DD title"), ...]`` — id 解決済みのみ
    (参議院で sid が逆引きできなかったものは含めない)。
    """
    today = date.today()
    start = args.date_from or ((args.date_to or today) - timedelta(days=7))
    end = args.date_to or today
    if start > end:
        start, end = end, start

    items: list[dict] = []
    if args.house in (None, "shugiin"):
        from .list import shugiin_list
        items.extend(shugiin_list.fetch_for_range(start, end))
    if args.house in (None, "sangiin"):
        from .list import sangiin_list
        items.extend(sangiin_list.fetch_for_range(start, end))

    out: list[tuple[str, str, str]] = []
    skipped_unresolved = 0
    for it in items:
        rid = it.get("id") or ""
        if not rid:
            skipped_unresolved += 1
            continue
        label = f"{it.get('date','')} {it.get('title','')}".strip()
        out.append((it.get("house", ""), rid, label))

    print(
        f"[fetch] 期間 {start} 〜 {end}: 列挙 {len(items)} 件 → fetch 対象 {len(out)} 件"
        + (f" (sid 未解決 {skipped_unresolved} 件は除外)" if skipped_unresolved else ""),
        file=sys.stderr,
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.target and args.date_from is None and args.date_to is None:
        parser.error("target または --from/--to のどちらかは指定してください")

    # 院ごとに振り分け
    sangiin_targets: list[str] = []
    shugiin_targets: list[str] = []
    for t in args.target:
        try:
            house = _classify(t)
        except ValueError as e:
            print(f"WARN: {e}", file=sys.stderr)
            continue
        if house == "sangiin":
            sangiin_targets.append(t)
        else:
            shugiin_targets.append(t)

    # --from/--to で列挙されたものを追加 (院は API 側で確定済みなので _classify を通さない)
    labels: dict[str, str] = {}
    if args.date_from is not None or args.date_to is not None:
        enumerated = _enumerate_by_date(args)
        existing = set(sangiin_targets) | set(shugiin_targets)
        for house, rid, label in enumerated:
            if rid in existing:
                continue
            existing.add(rid)
            labels[rid] = label
            if house == "sangiin":
                sangiin_targets.append(rid)
            else:
                shugiin_targets.append(rid)

    if args.dry_run:
        for t in sangiin_targets:
            tail = f"\t{labels[t]}" if t in labels else ""
            print(f"sangiin\t{t}{tail}")
        for t in shugiin_targets:
            tail = f"\t{labels[t]}" if t in labels else ""
            print(f"shugiin\t{t}{tail}")
        sys.exit(0)

    if not sangiin_targets and not shugiin_targets:
        print("[fetch] 取込対象が 0 件のため終了", file=sys.stderr)
        sys.exit(0)

    print(
        f"[fetch] 参議院: {len(sangiin_targets)} 件 / 衆議院: {len(shugiin_targets)} 件",
        file=sys.stderr,
    )

    exit_code = 0

    if sangiin_targets:
        from .sangiin.__main__ import main as sangiin_main
        argv_sangiin = list(sangiin_targets) + ["-o", args.output]
        if args.no_open:
            argv_sangiin.append("--no-open")
        if args.skip_if_done:
            argv_sangiin.append("--skip-if-done")
        saved = sys.argv
        sys.argv = ["sangiin"] + argv_sangiin
        try:
            try:
                sangiin_main()
            except SystemExit as e:
                if e.code:
                    exit_code = e.code
        finally:
            sys.argv = saved

    if shugiin_targets:
        from .shugiin.__main__ import main as shugiin_main
        argv_shugiin = list(shugiin_targets) + ["-o", args.output]
        if args.no_open:
            argv_shugiin.append("--no-open")
        if args.skip_if_done:
            argv_shugiin.append("--skip-if-done")
        if not args.asr:
            argv_shugiin.append("--no-asr")
        saved = sys.argv
        sys.argv = ["shugiin"] + argv_shugiin
        try:
            try:
                shugiin_main()
            except SystemExit as e:
                if e.code:
                    exit_code = e.code
        finally:
            sys.argv = saved

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
