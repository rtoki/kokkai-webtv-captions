"""自己完結 HTML レンダリング."""

from __future__ import annotations

import html as _html
import re
from pathlib import Path

from .detect import detect_turns, group_cues_by_speaker
from .extract import parse_vtt_cues, resolve_from_sangiin_detail


# ---------- データ構築 ----------

def build_sections(
    sangiin_url: str, vtt_path: Path, meta: dict | None = None
) -> dict:
    """発言者リスト・VTT から構造化されたセクションデータを生成。"""
    if meta is None:
        _, meta = resolve_from_sangiin_detail(sangiin_url)
    vtt = vtt_path.read_text(encoding="utf-8")
    cues = parse_vtt_cues(vtt)
    grouped = group_cues_by_speaker(cues, meta["speakers"])
    sections = []
    for g in grouped:
        sp = g["speaker"]
        turns = detect_turns(g["text"], sp["name"], sp.get("group", ""))
        sections.append({
            "start": sp["start"],
            "name": sp["name"],
            "group": sp["group"],
            "text": g["text"],
            "turns": turns,
        })
    return {"date": meta["date"], "title": meta["title"], "sections": sections}


# ---------- ユーティリティ ----------

def fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def slugify_id(i: int) -> str:
    return f"sp-{i+1:02d}"


def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[\s　]+", "_", s.strip())
    s = re.sub(r'[\\/:*?"<>|]+', "", s)
    return s[:max_len] or "output"


# ---------- 静的アセット ----------

CSS = """
:root {
  --bg: #faf9f7;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --accent: #154d74;
  --border: #d8d4cc;
  --questioner-bg: #f0f7fc; --questioner-bd: #154d74;
  --minister-bg: #fdf5e8;  --minister-bd: #b85c00;
  --sankoju-bg: #f4f0ed;   --sankoju-bd: #6b6b6b;
  --chair-bg: #f0f7e8;     --chair-bd: #4a7c3a;
}
* { box-sizing: border-box; }
html { scroll-padding-top: 80px; }
body {
  margin: 0; color: var(--fg); background: var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Yu Gothic UI", "Noto Sans JP", sans-serif;
  font-size: 15px; line-height: 1.75;
}
header.page {
  position: sticky; top: 0; z-index: 10;
  background: var(--bg); border-bottom: 1px solid var(--border);
  padding: 12px 24px;
}
header.page h1 { margin: 0 0 4px 0; font-size: 18px; }
header.page p.meta { margin: 0; color: var(--muted); font-size: 13px; }

.layout { display: grid; grid-template-columns: 280px 1fr; max-width: 1400px; margin: 0 auto; }

aside.toc {
  position: sticky; top: 70px; align-self: start;
  max-height: calc(100vh - 90px); overflow-y: auto;
  padding: 16px; border-right: 1px solid var(--border); font-size: 13px;
}
aside.toc h3 { margin: 0 0 8px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
aside.toc ol { list-style: none; padding: 0; margin: 0; }
aside.toc li { margin-bottom: 4px; }
aside.toc a {
  display: block; padding: 6px 8px;
  color: var(--fg); text-decoration: none; border-radius: 4px;
  border-left: 3px solid transparent;
}
aside.toc a:hover { background: #efece6; }
aside.toc a.active { background: #e9f0f6; border-left-color: var(--accent); font-weight: 600; }
aside.toc .toc-time { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
aside.toc .toc-group { color: var(--muted); font-size: 11px; }

main { padding: 24px 32px; min-width: 0; }

article.speaker {
  margin-bottom: 32px; padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
article.speaker > header h2 { margin: 0 0 4px; font-size: 17px; }
article.speaker > header .ts { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; margin-right: 8px; }
article.speaker > header .group { color: var(--muted); font-size: 13px; }

.turns { margin-top: 12px; display: flex; flex-direction: column; gap: 10px; }
.turn.section {
  padding: 10px 14px; border-radius: 6px;
  background: white; border-left: 3px solid var(--border);
}
.turn.turn-questioner { background: var(--questioner-bg); border-left-color: var(--questioner-bd); }
.turn.turn-minister   { background: var(--minister-bg);   border-left-color: var(--minister-bd); }
.turn.turn-bureaucrat { background: var(--sankoju-bg);    border-left-color: var(--sankoju-bd); }
.turn.turn-chair      { background: var(--chair-bg);      border-left-color: var(--chair-bd); }
.turn header {
  font-size: 12px; color: var(--muted); margin-bottom: 4px;
  display: flex; gap: 8px; align-items: center;
}
.turn .body { font-size: 14px; }
.turn .badge { display: inline-block; padding: 1px 8px; border-radius: 999px; background: rgba(0,0,0,0.06); font-size: 11px; color: var(--fg); }
.turn .badge-questioner { background: rgba(21,77,116,0.15); color: var(--questioner-bd); }
.turn .badge-minister   { background: rgba(184,92,0,0.15);  color: var(--minister-bd); }
.turn .badge-bureaucrat { background: rgba(107,107,107,0.18); }
.turn .badge-chair      { background: rgba(74,124,58,0.18); color: var(--chair-bd); }
.turn .who { font-weight: 600; color: var(--fg); }

.export-row { margin: 8px 0 4px; }
button.export {
  font-size: 12px; padding: 4px 10px; border: 1px solid var(--border);
  background: white; border-radius: 4px; cursor: pointer; color: var(--fg);
}
button.export:hover { background: #efece6; }

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  aside.toc {
    position: static; max-height: none;
    border-right: none; border-bottom: 1px solid var(--border);
  }
}
"""

JS = r"""
const links = document.querySelectorAll('aside.toc a');
const idToLink = {};
links.forEach(a => idToLink[a.getAttribute('href').slice(1)] = a);
const obs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(a => a.classList.remove('active'));
      const link = idToLink[e.target.id];
      if (link) link.classList.add('active');
    }
  });
}, { rootMargin: '-100px 0px -60% 0px' });
document.querySelectorAll('article.speaker').forEach(el => obs.observe(el));

function exportSection(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const blob = new Blob([el.innerText], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = id + '.txt'; a.click();
  URL.revokeObjectURL(url);
}
"""


# ---------- レンダリング ----------

def render_turn(t: dict) -> str:
    klass = t.get("klass", "questioner")
    role = t.get("role", "")
    speaker = _html.escape(t.get("speaker", ""))
    text = _html.escape(t["text"]).replace("\n", "<br>")
    role_label = _html.escape(role) if role else "発言"
    return (
        f'<div class="turn section turn-{klass}">'
        f'<header><span class="badge badge-{klass}">{role_label}</span>'
        f'<span class="who">{speaker}</span></header>'
        f'<div class="body">{text}</div></div>'
    )


def render_speaker(sec: dict, i: int) -> str:
    sid = slugify_id(i)
    turns_html = "\n".join(render_turn(t) for t in sec["turns"]) or \
        '<div class="placeholder">本文未抽出</div>'
    return f"""<article class="speaker" id="{sid}">
  <header>
    <h2><span class="ts">{fmt_time(sec['start'])}</span>{_html.escape(sec['name'])}</h2>
    <div class="group">{_html.escape(sec['group'])}</div>
    <div class="export-row"><button class="export" onclick="exportSection('{sid}')">この発言をエクスポート</button></div>
  </header>
  <div class="turns">{turns_html}</div>
</article>"""


def render_toc(sections: list[dict]) -> str:
    items = []
    for i, sec in enumerate(sections):
        sid = slugify_id(i)
        items.append(
            f'<li><a href="#{sid}">'
            f'<span class="toc-time">{fmt_time(sec["start"])}</span> '
            f'{_html.escape(sec["name"])}'
            f'<div class="toc-group">{_html.escape(sec["group"])}</div>'
            f'</a></li>'
        )
    return "<ol>" + "\n".join(items) + "</ol>"


_HOUSE_BADGE_CSS = """
header.page .badge-house {
  display: inline-block; margin-left: 6px;
  padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
}
header.page .badge-house.badge-sangiin { background: #e0ecf7; color: #284c75; }
header.page p.pipeline {
  margin: 6px 0 0; color: var(--muted); font-size: 12px;
  display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap;
}
header.page p.pipeline .pipeline-tag {
  background: #ece6e0; color: #5c4a2a;
  padding: 1px 6px; border-radius: 4px; font-size: 11px; font-weight: 600;
}
header.page p.pipeline .pipeline-note { color: var(--muted); }
"""


def render_html(data: dict) -> str:
    sections = data["sections"]
    title = f"{data['date']} {data['title']}"
    pipeline_block = (
        '  <p class="pipeline">'
        '<span class="pipeline-tag">公式 VTT 字幕</span>'
        '<span class="pipeline-note">人手による字幕 (AI 校正なし)</span>'
        '</p>\n'
    )
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>{CSS}{_HOUSE_BADGE_CSS}</style>
</head>
<body>
<header class="page">
  <h1>{_html.escape(data['title'])} <span class="badge-house badge-sangiin">参議院</span></h1>
  <p class="meta">{_html.escape(data['date'])}・発言者 {len(sections)}名</p>
{pipeline_block}</header>
<div class="layout">
  <aside class="toc">
    <h3>目次</h3>
    {render_toc(sections)}
  </aside>
  <main>
    {chr(10).join(render_speaker(s, i) for i, s in enumerate(sections))}
  </main>
</div>
<script>{JS}</script>
</body>
</html>
"""
