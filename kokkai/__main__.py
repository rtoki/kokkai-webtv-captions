"""統合 CLI: ``kkcap <subcommand> [args...]``

サブコマンド:
  sangiin   参議院 webtv の AI 字幕から発言者別 HTML を生成
  shugiin   衆議院 shugiintv の音声からローカル ASR で文字起こし
  search    out/ 配下に溜まった文字起こしを BM25 で全文検索
  list      両院の中継一覧を表示 (取込済 / 未取込を識別)
  fetch     複数の sid / deli_id を一括取込 (院は自動判定)

各 subcommand は ``kokkai.<name>.__main__:main`` を呼び出すだけのディスパッチャ。
``kkcap-search`` / ``kkcap-list`` / ``kkcap-fetch`` 単独コマンドも `kkcap <sub>` と等価。
"""

from __future__ import annotations

import argparse
import sys


SUBCOMMANDS = {
    "sangiin": "kokkai.sangiin.__main__",
    "shugiin": "kokkai.shugiin.__main__",
    "search":  "kokkai.search.__main__",
    "list":    "kokkai.list.__main__",
    "fetch":   "kokkai.fetch",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        sys.exit(0 if (argv and argv[0] in ("-h", "--help", "help")) else 2)

    if argv[0] == "--version":
        from . import __version__
        print(f"kokkai-webtv-captions {__version__}")
        sys.exit(0)

    sub = argv[0]
    if sub not in SUBCOMMANDS:
        print(f"unknown subcommand: {sub!r}", file=sys.stderr)
        _print_help()
        sys.exit(2)

    # 該当 subcommand の main(args) を呼び出す
    mod_name = SUBCOMMANDS[sub]
    import importlib
    mod = importlib.import_module(mod_name)
    # 各 subcommand の main() は sys.argv を参照する作りなので、書き換えてから呼ぶ
    saved = sys.argv
    try:
        sys.argv = [sub] + argv[1:]
        mod.main()
    finally:
        sys.argv = saved


def _print_help() -> None:
    print(
        "usage: kkcap <subcommand> [args...]\n"
        "\n"
        "発見系 (sid / deli_id を知らなくても OK):\n"
        "  list [--date Y-M-D]     両院の中継一覧を表示 (取込済 / 未取込を識別)\n"
        "  search <query>          out/ の文字起こしを BM25 で全文検索\n"
        "\n"
        "取得系:\n"
        "  fetch <id> [<id>...]    複数の sid / deli_id を一括取込 (院は自動判定)\n"
        "  sangiin <sid>...        参議院 webtv → 発言者別 HTML\n"
        "  shugiin <deli_id>...    衆議院 shugiintv → タイムライン / ASR 字幕 HTML\n"
        "\n"
        "各サブコマンドの詳細は `kkcap <subcommand> --help` を参照。\n"
        "\n"
        "オプション:\n"
        "  --version               バージョン表示\n"
        "  -h, --help              このヘルプ\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
