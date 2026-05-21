"""
複数 sid に対する発言者検出のスモークテスト/トレーニング用ランナー.

各 sid についてキャッシュ済み VTT を読み、検出統計を出力する。

使い方:
    python -m tests.run_smoke 8500 8955 ...

事前に ``python -m kokkai.sangiin <sid>`` で VTT を取得しておく必要あり。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from kokkai.sangiin.detect import detect_turns, group_cues_by_speaker
from kokkai.sangiin.extract import parse_vtt_cues, resolve_from_sangiin_detail


SANGIIN_URL_TMPL = "https://www.webtv.sangiin.go.jp/webtv/detail.php?sid={}"


def smoke_one(sid: str, out_dir: Path) -> dict:
    """指定 sid の VTT を読み、検出統計を返す。"""
    url = SANGIIN_URL_TMPL.format(sid)
    _, meta = resolve_from_sangiin_detail(url)
    base = f"{meta['date']}_{meta['title']}".replace("/", "_").replace(" ", "_")
    vtt_path = out_dir / f"{base}.vtt"
    if not vtt_path.exists():
        return {
            "sid": sid,
            "error": f"VTT 未取得 ({vtt_path}) — 先に `python -m sangiin {sid}` を実行",
        }

    vtt = vtt_path.read_text(encoding="utf-8")
    cues = parse_vtt_cues(vtt)
    speakers = meta.get("speakers") or []
    if not speakers:
        return {"sid": sid, "error": "発言者リスト無し"}

    grouped = group_cues_by_speaker(cues, speakers)
    role_counter: Counter[str] = Counter()
    klass_counter: Counter[str] = Counter()
    zero_turn_speakers: list[str] = []
    detected_speakers: set[str] = set()
    for g in grouped:
        sp = g["speaker"]
        turns = detect_turns(g["text"], sp["name"], sp.get("group", ""))
        non_q = [t for t in turns if t["klass"] != "questioner"]
        if len(g["text"]) > 200 and not non_q:
            # 200字超ある質問者セクションで政府側ターンが0なのは怪しい
            zero_turn_speakers.append(sp["name"])
        for t in turns:
            role_counter[t["role"]] += 1
            klass_counter[t["klass"]] += 1
            if t["klass"] != "questioner":
                detected_speakers.add(t["speaker"])

    return {
        "sid": sid,
        "date": meta["date"],
        "title": meta["title"],
        "n_questioners": len(speakers),
        "n_cues": len(cues),
        "klass": dict(klass_counter),
        "roles": dict(role_counter.most_common(20)),
        "detected_respondents": sorted(detected_speakers),
        "zero_turn_in_long_section": zero_turn_speakers,
    }


def print_report(r: dict) -> None:
    if r.get("error"):
        print(f"\n# {r['sid']}: ERROR {r['error']}")
        return
    print(f"\n# sid={r['sid']}: {r['date']} {r['title']}")
    print(f"  字幕cue数: {r['n_cues']}, 質問者: {r['n_questioners']}名")
    print(f"  分類: {r['klass']}")
    print(f"  検出された答弁者 ({len(r['detected_respondents'])}名): {', '.join(r['detected_respondents']) or '(なし)'}")
    print(f"  役職 Top: {r['roles']}")
    if r["zero_turn_in_long_section"]:
        print(f"  ⚠ 政府側ターン0の質問者: {r['zero_turn_in_long_section']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sids", nargs="+", help="sid 番号 (複数指定可)")
    parser.add_argument("-o", "--output", default="out", help="VTT キャッシュディレクトリ")
    args = parser.parse_args()

    out_dir = Path(args.output)
    targets = list(args.sids)

    any_fail = False
    for t in targets:
        try:
            r = smoke_one(t, out_dir)
        except Exception as e:  # noqa
            print(f"\n# {t}: EXCEPTION {e}", file=sys.stderr)
            any_fail = True
            continue
        print_report(r)
        if r.get("error"):
            any_fail = True
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
