"""tests/fixtures/ にローカル fixture HTML を取得する開発者向けスクリプト。

外部サイト由来の HTML はリポジトリに含めない方針のため、テストで参照する
fixture は各開発者がローカルで本スクリプトを実行して用意する。
fixture が存在しないテストは自動 skip される。

使い方:
    python scripts/fetch_fixtures.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


FIXTURES = {
    "shugiin_deli56246.html": (
        "https://www.shugiintv.go.jp/jp/index.php?ex=VL&deli_id=56246&media_type="
    ),
    "shugiin_kaiha_011.html": (
        "https://www.shugiin.go.jp/internet/itdb_annai.nsf/html/statics/syu/011kaiha.htm"
    ),
}

HEADERS = {
    "User-Agent": "kokkai-webtv-captions/0.2.0",
    "Referer": "https://www.shugiin.go.jp/",
}


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in FIXTURES.items():
        dest = out_dir / name
        if dest.exists():
            print(f"skip (exists): {dest.relative_to(out_dir.parent.parent)}")
            continue
        print(f"fetch: {url}")
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  -> {dest.relative_to(out_dir.parent.parent)} ({len(data)} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
