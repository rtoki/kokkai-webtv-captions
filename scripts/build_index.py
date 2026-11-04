#!/usr/bin/env python3
"""out/*.meta.json と *.html を dist/ に stage し、dist/index.html を生成する。

dist/ がそのまま Cloudflare Pages へのデプロイ対象になる。
out/ は wav / transcript.json / その他中間ファイルも含むので直接デプロイすると
25 MiB 上限に引っかかるため、公開対象だけを dist/ にコピーする方式を取る。
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
DIST_DIR = ROOT / "dist"

# 参議院 meta.json は date を "2026年5月22日" 形式で持つので衆議院 (ISO) に揃える。
_JP_DATE = re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$")


def normalize_date(s: str) -> str:
    m = _JP_DATE.match(s.strip())
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return s.strip()


def find_html(meta_path: Path) -> Path | None:
    base = meta_path.name[: -len(".meta.json")]
    for candidate in (f"{base}_asr.html", f"{base}.html"):
        p = meta_path.parent / candidate
        if p.exists():
            return p
    return None


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for meta in sorted(OUT_DIR.glob("*.meta.json")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        html = find_html(meta)
        if html is None:
            continue
        rows.append(
            {
                "date": normalize_date(str(data.get("date", ""))),
                "title": str(data.get("title", "")),
                "house": str(data.get("house", "")),
                "id": str(data.get("id", "")),
                "href": html.name,
                "meta_href": meta.name,
                "speakers": len(data.get("speakers", []) or []),
            }
        )
    return rows


HTML_HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>国会 議事録 (kkcap)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<style>
:root{color-scheme:light dark}
body{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Hiragino Sans",sans-serif;
max-width:880px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.65}
h1{font-size:1.4rem;border-bottom:1px solid #ddd;padding-bottom:.4rem}
h2{font-size:1.0rem;margin:1.8rem 0 .3rem;color:#555;font-weight:600}
ul{list-style:none;padding-left:0;margin:.2rem 0}
li{padding:.35rem 0;border-bottom:1px dotted #eee;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
a{color:#06c;text-decoration:none} a:hover{text-decoration:underline}
.badge{display:inline-block;font-size:.7rem;padding:.1rem .45rem;border-radius:3px;color:#fff;flex:none}
.s-sa{background:#1976d2} .s-sh{background:#388e3c}
.title{flex:1;min-width:14rem}
.meta{color:#888;font-size:.78rem}
.meta a{color:#888}
header p{color:#555;font-size:.9rem}
footer{margin-top:3rem;color:#888;font-size:.78rem;border-top:1px solid #eee;padding-top:1rem}
@media (prefers-color-scheme:dark){
  body{background:#111;color:#ddd}
  h1{border-color:#333} h2{color:#aaa} li{border-color:#222}
  a{color:#6ab7ff} footer{border-color:#222}
}
</style></head><body>
<header>
<h1>国会 議事録</h1>
<p>会議 __COUNT__ 件。
<a href="https://github.com/rtoki/kokkai-webtv-captions">kokkai-webtv-captions</a> によるローカル生成版。
正式な記録は <a href="https://kokkai.ndl.go.jp/">国会会議録検索システム</a> を参照してください。</p>
</header>
"""

HTML_FOOT = """<footer>
本サイトは非公式の補助ツールによる文字起こし版です。
ASR/LLM 校正には誤認識が含まれます。
参議院 <a href="https://www.webtv.sangiin.go.jp/webtv/howto.php">利用方法</a> /
衆議院 <a href="https://www.shugiintv.go.jp/jp/index.php?ex=CR">著作権・リンクについて</a>。
</footer></body></html>
"""


def render(rows: list[dict]) -> str:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)

    parts = [HTML_HEAD.replace("__COUNT__", str(len(rows)))]
    for date in sorted(by_date.keys(), reverse=True):
        items = sorted(by_date[date], key=lambda r: (r["house"], r["title"]))
        parts.append(f"<h2>{escape(date)}</h2><ul>")
        for r in items:
            badge_cls = "s-sa" if r["house"] == "sangiin" else "s-sh"
            badge_lbl = "参" if r["house"] == "sangiin" else "衆"
            parts.append(
                f'<li><span class="badge {badge_cls}">{badge_lbl}</span>'
                f'<span class="title"><a href="{escape(r["href"])}">{escape(r["title"])}</a></span>'
                f'<span class="meta">発言者 {r["speakers"]} 名 · '
                f'<a href="{escape(r["meta_href"])}">meta</a></span></li>'
            )
        parts.append("</ul>")
    parts.append(HTML_FOOT)
    return "\n".join(parts)


def stage_to_dist(rows: list[dict]) -> None:
    """dist/ を作り直して out/ から *.html と *.meta.json をコピーする。"""
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)
    for r in rows:
        for name in (r["href"], r["meta_href"]):
            src = OUT_DIR / name
            dst = DIST_DIR / name
            shutil.copy2(src, dst)


ROBOTS_TXT = """User-agent: *
Disallow: /
"""


def main() -> None:
    rows = load_rows()
    stage_to_dist(rows)
    (DIST_DIR / "index.html").write_text(render(rows), encoding="utf-8")
    (DIST_DIR / "robots.txt").write_text(ROBOTS_TXT, encoding="utf-8")
    print(f"staged {len(rows)} meetings + index.html + robots.txt into {DIST_DIR}")


if __name__ == "__main__":
    main()
