"""
衆議院モードの HTML レンダリング (Phase 2: 字幕無しタイムライン版)。

Phase 3 で ASR テキストが入ったら、本モジュールに ``render_html_with_asr()``
を追加して切り替える想定。CSS/JS は ``kokkai.sangiin.render`` と将来共通化したい
が、Phase 2 ではコピー流用でスコープを絞る。
"""

from __future__ import annotations

import html as _html


def fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def slugify_id(i: int) -> str:
    return f"sp-{i + 1:02d}"


def _classify(group: str) -> str:
    """所属文字列から CSS クラスを決定。委員長は緑、それ以外は青系。"""
    if "委員長" in group or "議長" in group or "副議長" in group:
        return "chair"
    return "questioner"


CSS = """
:root {
  --bg: #faf9f7;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --accent: #154d74;
  --border: #d8d4cc;
  --questioner-bg: #f0f7fc; --questioner-bd: #154d74;
  --chair-bg: #f0f7e8;      --chair-bd: #4a7c3a;
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
header.page .badge-phase {
  display: inline-block; margin-left: 8px;
  background: #fdf3d8; color: #8a6b1a;
  padding: 1px 8px; border-radius: 999px; font-size: 11px;
}
header.page .badge-house {
  display: inline-block; margin-left: 6px;
  padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
}
header.page .badge-house.badge-shugiin { background: #fce6e6; color: #903030; }
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
  margin-bottom: 24px; padding: 16px 18px;
  background: white;
  border-radius: 6px; border: 1px solid var(--border);
  border-left: 3px solid var(--questioner-bd);
}
article.speaker.chair { border-left-color: var(--chair-bd); background: var(--chair-bg); }
article.speaker > header h2 { margin: 0 0 4px; font-size: 17px; }
article.speaker > header .ts { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; margin-right: 8px; }
article.speaker > header .duration { color: var(--muted); font-size: 12px; margin-left: 8px; }
article.speaker > header .group { color: var(--muted); font-size: 13px; }

.actions { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.actions a, .actions button {
  font-size: 12px; padding: 4px 10px; border: 1px solid var(--border);
  background: white; border-radius: 4px; cursor: pointer;
  color: var(--fg); text-decoration: none;
}
.actions a:hover, .actions button:hover { background: #efece6; }

.placeholder {
  margin-top: 10px; padding: 10px 14px; border-radius: 4px;
  background: #fff8e6; border-left: 3px solid #e6c068;
  color: #6b5d2c; font-size: 13px;
}

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
if (links.length > 0) {
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
}
"""


def _annotate_durations(speakers: list[dict]) -> list[dict]:
    """各発言者に end (= 次の発言者の start) を付加。最後の発言者は end=None。"""
    out = []
    for i, sp in enumerate(speakers):
        nxt = speakers[i + 1]["start"] if i + 1 < len(speakers) else None
        out.append({**sp, "end": nxt})
    return out


def _render_speaker_card(sp: dict, i: int, deli_id: str, page_url: str) -> str:
    sid = slugify_id(i)
    klass = _classify(sp.get("group", ""))
    duration_html = ""
    if sp.get("end") is not None:
        dur_sec = int(sp["end"] - sp["start"])
        m, s = divmod(dur_sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            dur = f"{h}時間{m}分"
        elif m > 0:
            dur = f"{m}分{s}秒"
        else:
            dur = f"{s}秒"
        duration_html = f'<span class="duration">所要 {dur}</span>'
    jump_url = f"{page_url}&time={sp['start']}"
    return f"""<article class="speaker {klass}" id="{sid}">
  <header>
    <h2>
      <span class="ts">{fmt_time(sp['start'])}</span>{_html.escape(sp['name'])}
      {duration_html}
    </h2>
    <div class="group">{_html.escape(sp.get('group', ''))}</div>
    <div class="actions">
      <a href="{_html.escape(jump_url)}" target="_blank" rel="noopener">▶ 衆議院ページで再生</a>
    </div>
  </header>
  <div class="placeholder">
    発言テキストは現在未対応 (Phase 3 で faster-whisper turbo + 公式ページ語彙 hint による文字起こしを追加予定)。
    上記リンクから衆議院公式ページに飛ぶと、この時刻から動画が再生されます。
  </div>
</article>"""


# ---------- Phase 3: ASR 出力統合版 ----------

def _format_cue_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def _render_cue(cue: dict, speaker_start: float) -> str:
    # cue 時刻を発言者開始からの相対秒で表示 (絶対時刻だと長い)
    rel = max(0.0, cue["start"] - speaker_start)
    return (
        f'<div class="cue">'
        f'<span class="cue-time">+{_format_cue_time(rel)}</span>'
        f'<span class="cue-text">{_html.escape(cue["text"])}</span>'
        f'</div>'
    )


def _render_speaker_card_with_asr(
    group: dict,
    i: int,
    deli_id: str,
    page_url: str,
) -> str:
    """assign_cues_to_speakers の出力を 1 カードに描画。"""
    sid = slugify_id(i)
    klass = _classify(group.get("group", ""))
    cues = group.get("cues", [])
    duration_html = ""
    if group.get("end") is not None:
        dur_sec = int(group["end"] - group["start"])
        m, s = divmod(dur_sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            dur = f"{h}時間{m}分"
        elif m > 0:
            dur = f"{m}分{s}秒"
        else:
            dur = f"{s}秒"
        duration_html = f'<span class="duration">所要 {dur}</span>'
    jump_url = f"{page_url}&time={group['start']}"

    if cues:
        body = (
            '<div class="cues">'
            + "\n".join(_render_cue(c, group["start"]) for c in cues)
            + "</div>"
        )
    else:
        body = (
            '<div class="placeholder">'
            'この発言者の時間範囲に cue が見つかりませんでした。'
            '</div>'
        )

    return f"""<article class="speaker {klass}" id="{sid}">
  <header>
    <h2>
      <span class="ts">{fmt_time(group['start'])}</span>{_html.escape(group['name'])}
      {duration_html}
    </h2>
    <div class="group">{_html.escape(group.get('group', ''))}</div>
    <div class="actions">
      <a href="{_html.escape(jump_url)}" target="_blank" rel="noopener">▶ 衆議院ページで再生</a>
      <button class="export" onclick="exportSection('{sid}')">エクスポート</button>
    </div>
  </header>
  {body}
</article>"""


def _render_toc(speakers: list[dict]) -> str:
    items = []
    for i, sp in enumerate(speakers):
        sid = slugify_id(i)
        items.append(
            f'<li><a href="#{sid}">'
            f'<span class="toc-time">{fmt_time(sp["start"])}</span> '
            f'{_html.escape(sp["name"])}'
            f'<div class="toc-group">{_html.escape(sp.get("group", ""))}</div>'
            f'</a></li>'
        )
    return "<ol>" + "\n".join(items) + "</ol>"


def render_pipeline_summary(pipeline: dict | None) -> str:
    """meta.json の ``pipeline`` セクションを 1 行 HTML テキストにする。

    どの校正パイプラインで生成された HTML かをヘッダで一目分かるようにするための
    生成。空文字なら表示しない (古い meta.json の互換)。
    """
    if not pipeline:
        return ""
    phase = pipeline.get("phase", "")
    if phase == "vtt":
        return ('<span class="pipeline-tag">公式 VTT 字幕</span>'
                '<span class="pipeline-note">人手による字幕 (AI 校正なし)</span>')
    if phase == "phase2":
        return ('<span class="pipeline-tag">タイムラインのみ</span>'
                '<span class="pipeline-note">ASR 未実行 (発言者リスト表示のみ)</span>')
    if phase != "asr":
        return ""
    bits: list[str] = []
    backend = pipeline.get("asr_backend") or "?"
    model = pipeline.get("asr_model") or "?"
    bits.append(f"ASR: {_html.escape(backend)} / {_html.escape(str(model))}")
    if pipeline.get("hint"):
        chars = pipeline.get("hint_chars", 0)
        bits.append(f"hint {chars}字注入")
    else:
        bits.append("hint なし")
    n_h = pipeline.get("preclean_hallucinations") or 0
    n_l = pipeline.get("preclean_loops") or 0
    if n_h or n_l:
        parts = []
        if n_h:
            parts.append(f"幻覚 {n_h} 件")
        if n_l:
            parts.append(f"degenerate loop {n_l} 件")
        bits.append("preclean " + "・".join(parts))
    if pipeline.get("glossary"):
        n = pipeline.get("glossary_changes") or 0
        bits.append(f"glossary {n} 箇所")
    if pipeline.get("llm_context"):
        terms = pipeline.get("llm_context_terms") or []
        bits.append(f"2-pass context ({len(terms)} 語)")
    if pipeline.get("llm_correct"):
        n = pipeline.get("llm_correct_changes") or 0
        backend = pipeline.get("llm_backend") or "?"
        bits.append(f"LLM 校正 {n} 箇所 ({_html.escape(backend)})")
    else:
        bits.append("LLM 校正なし")
    summary = "・".join(bits)
    return f'<span class="pipeline-tag">AI パイプライン</span><span class="pipeline-note">{summary}</span>'


def render_asr_html(
    meta: dict, speaker_groups: list[dict], pipeline: dict | None = None,
) -> str:
    """
    ASR 出力統合版 HTML を生成 (Phase 3 出力)。

    Args:
        meta: extract.parse_meta_html の結果
        speaker_groups: asr.assign_cues_to_speakers の結果
        pipeline: __main__.py で組み立てた pipeline dict (ヘッダ表示用)
    """
    groups = _annotate_durations(speaker_groups)
    title = f"{meta['date']} {meta['title']}".strip()
    title_for_head = title or f"shugiin deli_id={meta['deli_id']}"
    total_cues = sum(len(g.get("cues", [])) for g in groups)

    if not groups:
        body_main = '<div class="placeholder">発言者リストが取得できませんでした。</div>'
        toc_html = ""
    else:
        body_main = "\n".join(
            _render_speaker_card_with_asr(g, i, meta["deli_id"], meta["page_url"])
            for i, g in enumerate(groups)
        )
        toc_html = f'<aside class="toc"><h3>目次</h3>{_render_toc(groups)}</aside>'

    pipeline_html = render_pipeline_summary(pipeline)
    pipeline_block = f'  <p class="pipeline">{pipeline_html}</p>\n' if pipeline_html else ""

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{_html.escape(title_for_head)}</title>
<style>{CSS}{_ASR_CSS}</style>
</head>
<body>
<header class="page">
  <h1>{_html.escape(meta.get('title', ''))} <span class="badge-house badge-shugiin">衆議院</span> <span class="badge-phase badge-asr">ASR 字幕付き (Phase 3)</span></h1>
  <p class="meta">
    {_html.escape(meta.get('date', ''))}・発言者 {len(groups)}名・cue {total_cues}件・
    <a href="{_html.escape(meta['page_url'])}" target="_blank" rel="noopener">衆議院公式ページ</a>
  </p>
{pipeline_block}</header>
<div class="layout">
  {toc_html}
  <main>
    {body_main}
  </main>
</div>
<script>{JS}{_ASR_JS}</script>
</body>
</html>
"""


_ASR_CSS = """
.cues { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
.cue {
  display: flex; gap: 10px; align-items: baseline;
  padding: 4px 0; font-size: 14px;
}
.cue-time {
  color: var(--muted); font-size: 11px;
  font-variant-numeric: tabular-nums;
  min-width: 40px; flex-shrink: 0;
}
.cue-text { flex: 1; }
.badge-phase.badge-asr { background: #e6f3df; color: #4a7c3a; }
button.export {
  font-size: 12px; padding: 4px 10px; border: 1px solid var(--border);
  background: white; border-radius: 4px; cursor: pointer; color: var(--fg);
}
button.export:hover { background: #efece6; }
"""

_ASR_JS = r"""
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


def render_timeline_html(meta: dict, pipeline: dict | None = None) -> str:
    """字幕無しの「発言者タイムライン」HTML を生成 (Phase 2 出力)。"""
    speakers = _annotate_durations(meta.get("speakers", []))
    title = f"{meta['date']} {meta['title']}".strip()
    title_for_head = title or f"shugiin deli_id={meta['deli_id']}"

    if not speakers:
        body_main = (
            '<div class="placeholder">'
            'この動画には発言者一覧が登録されていません。'
            '</div>'
        )
        toc_html = ""
    else:
        body_main = "\n".join(
            _render_speaker_card(sp, i, meta["deli_id"], meta["page_url"])
            for i, sp in enumerate(speakers)
        )
        toc_html = f'<aside class="toc"><h3>目次</h3>{_render_toc(speakers)}</aside>'

    pipeline_html = render_pipeline_summary(pipeline or {"phase": "phase2"})
    pipeline_block = f'  <p class="pipeline">{pipeline_html}</p>\n' if pipeline_html else ""

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{_html.escape(title_for_head)}</title>
<style>{CSS}</style>
</head>
<body>
<header class="page">
  <h1>{_html.escape(meta.get('title', ''))} <span class="badge-house badge-shugiin">衆議院</span> <span class="badge-phase">字幕なし (Phase 2)</span></h1>
  <p class="meta">
    {_html.escape(meta.get('date', ''))}・発言者 {len(speakers)}名・
    <a href="{_html.escape(meta['page_url'])}" target="_blank" rel="noopener">衆議院公式ページ</a>
  </p>
{pipeline_block}</header>
<div class="layout">
  {toc_html}
  <main>
    {body_main}
  </main>
</div>
<script>{JS}</script>
</body>
</html>
"""
