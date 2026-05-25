"""
話者分離 POC: cluster ID 付き cue から発言者単位 HTML を生成する。

入力:
- cues_with_speaker.json (asr_poc_diarize.py の出力。各 cue が speaker_id を持つ)
- meta.json (公式メタ。speakers の議員時刻と照合して cluster の素性を推定)

cluster の名前付け規則:

  1. **議員枠の支配的 cluster** (発話時間が最長) → その議員本人と同定
  2. **複数議員枠に跨る cluster** (3 枠以上 出現) → "(答弁者/委員長候補)"
     (大臣のように複数議員に答弁を繰り返す共通話者を捉える)
  3. **非支配 cluster** で発話量が小さい (cue 数 / 累積秒数が閾値未満)
     → **支配的 cluster に統合** (同じ議員の声の揺らぎを別人と誤識別したものを救う)
  4. **非支配 cluster** で発話量が十分 → その議員枠の "(○○ 枠内答弁者)"
  5. テキスト検出した答弁者 (kokkai.shugiin.answerer.inject_answerer_turns) が
     cluster の時刻範囲内に存在 → その役職と名前を cluster に割り当て (音響 +
     テキストの統合)

既知の限界 (忖度なし):
- speaker_id の安定性は会議音声の質 (マイク位置 / 同時発話) に依存。複数 cluster
  が同じ話者の声の揺らぎを別人と誤識別するケースは依然残る (filter してもゼロ
  にならない)。
- 議員 1 名分の声がチャンク毎に異なる cluster に分かれる場合、長尺会議では特に
  起きやすい (sherpa-onnx の embedding clustering の特性)。
- 「(答弁者/委員長候補)」/「(○○ 枠内答弁者)」のラベルはあくまでヒューリスティック
  で、実際の役職とは限らない。検証は HTML 内の発言テキストを目視するのが現状の
  唯一の確認方法。

出力:
- out_poc/diarize/<stem>_diarize.html (発言者ごとの article、cluster ベース)
- out_poc/diarize/<stem>_cluster_map.json (cluster → name/role のマッピング、
  debug/inspection 用)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _identify_common_answerer_clusters(
    cues: list[dict],
    official_speakers: list[dict],
    *,
    min_slots: int = 3,
) -> set[int]:
    """N 個以上の議員枠に跨がって出現する cluster を「共通答弁者 (大臣・参考人)」
    候補として返す。

    これを先に確定して dominant 計算から除外しないと、答弁者 cluster が最初の
    議員枠の primary に取られて、その議員自身の cluster が「○○ 枠内答弁者」扱い
    に化ける。
    """
    appearance = _cluster_appearance_slots(cues, official_speakers)
    return {cid for cid, slots in appearance.items() if len(slots) >= min_slots}


def _identify_dominant_cluster_per_member_slot(
    cues: list[dict],
    official_speakers: list[dict],
    *,
    exclude_clusters: set[int] | None = None,
) -> dict[int, dict]:
    """公式議員の時間枠ごとに、支配的 cluster (最も発話時間が長い cluster) を
    その議員本人と同定する。

    ``exclude_clusters`` に共通答弁者 cluster を渡すと、それらは dominant 候補から
    除外される。複数枠で答弁する大臣 cluster が最初の議員枠を奪うのを防ぐ。

    Returns:
        {cluster_id: {"matched_speaker": {...議員 dict...}, "dominance": "primary"}}
        primary とマッチしなかった cluster は呼び出し側で別扱い (議員枠内の非議員話者)。
    """
    exclude = exclude_clusters or set()
    sorted_sp = sorted(official_speakers, key=lambda s: s["start"])
    slots: list[tuple[float, float, dict]] = []
    for i, s in enumerate(sorted_sp):
        end = (
            sorted_sp[i + 1]["start"] if i + 1 < len(sorted_sp) else float("inf")
        )
        slots.append((s["start"], end, s))

    # 各議員枠内の cluster ごと発話時間 を集計 (答弁者 cluster は除外)
    dominant_by_slot: list[tuple[int | None, float]] = []
    for s_start, s_end, s in slots:
        cluster_time: dict[int, float] = {}
        for c in cues:
            sp = c.get("speaker_id")
            if sp is None or sp in exclude:
                continue
            cs = float(c["start"])
            ce = float(c.get("end", cs))
            ov_start = max(cs, s_start)
            ov_end = min(ce, s_end)
            ov = max(0.0, ov_end - ov_start)
            if ov > 0:
                cluster_time[sp] = cluster_time.get(sp, 0.0) + ov
        if not cluster_time:
            dominant_by_slot.append((None, 0.0))
            continue
        sp_id, t = max(cluster_time.items(), key=lambda x: x[1])
        dominant_by_slot.append((sp_id, t))

    out: dict[int, dict] = {}
    for (sp_id, _t), (_, _, official) in zip(dominant_by_slot, slots):
        if sp_id is None or sp_id in out:
            # 既に他の議員枠でその cluster が支配的だった場合は最初の議員に固定
            continue
        out[sp_id] = {"matched_speaker": official, "dominance": "primary"}
    return out


def _cluster_appearance_slots(
    cues: list[dict],
    official_speakers: list[dict],
) -> dict[int, set[int]]:
    """各 cluster が「どの議員枠」に出現したかの index 集合を返す。

    複数の議員枠に同じ cluster が出現 → 答弁を繰り返す共通話者 (大臣・政府参考人・委員長)
    と推定できる。
    """
    sorted_sp = sorted(official_speakers, key=lambda s: s["start"])
    slots: list[tuple[float, float]] = []
    for i, s in enumerate(sorted_sp):
        end = (
            sorted_sp[i + 1]["start"] if i + 1 < len(sorted_sp) else float("inf")
        )
        slots.append((s["start"], end))

    appearance: dict[int, set[int]] = {}
    for c in cues:
        sp = c.get("speaker_id")
        if sp is None:
            continue
        cs = float(c["start"])
        ce = float(c.get("end", cs))
        for idx, (s_start, s_end) in enumerate(slots):
            if ce > s_start and cs < s_end:
                appearance.setdefault(sp, set()).add(idx)
    return appearance


def _cluster_speech_stats(cues: list[dict]) -> dict[int, dict]:
    """cluster ごとの発話統計 (cue 数 / 累積秒数 / 範囲) を返す。"""
    stats: dict[int, dict] = {}
    for c in cues:
        sp = c.get("speaker_id")
        if sp is None:
            continue
        cs = float(c["start"])
        ce = float(c.get("end", cs))
        if sp not in stats:
            stats[sp] = {
                "cue_count": 0,
                "total_duration": 0.0,
                "first_start": cs,
                "last_end": ce,
            }
        stats[sp]["cue_count"] += 1
        stats[sp]["total_duration"] += max(0.0, ce - cs)
        stats[sp]["first_start"] = min(stats[sp]["first_start"], cs)
        stats[sp]["last_end"] = max(stats[sp]["last_end"], ce)
    return stats


def _merge_tiny_clusters_into_slot_dominant(
    cues: list[dict],
    official_speakers: list[dict],
    *,
    min_cues: int,
    min_duration: float,
    exclude_clusters: set[int] | None = None,
) -> list[dict]:
    """非支配 cluster で発話量が閾値未満のものを、その cue が属する議員枠の
    支配的 cluster に書き換える (同じ議員の音響的揺らぎを救う)。

    ``exclude_clusters`` (共通答弁者 cluster) は merge 先の候補から除外する。
    これがないと、答弁者 cluster が slot 内で最も発話時間が長いケースで、
    短い質問者 cluster がそこへ吸収されてしまう。

    返り値は新 cue 列 (元 cue は変更しない)。
    """
    if not official_speakers:
        return cues
    exclude = exclude_clusters or set()
    sorted_sp = sorted(official_speakers, key=lambda s: s["start"])
    # 議員枠の [start, end) を計算
    slot_ranges: list[tuple[float, float, int]] = []
    for i, _s in enumerate(sorted_sp):
        end = (
            sorted_sp[i + 1]["start"] if i + 1 < len(sorted_sp) else float("inf")
        )
        slot_ranges.append((sorted_sp[i]["start"], end, i))

    dominant = _identify_dominant_cluster_per_member_slot(
        cues, official_speakers, exclude_clusters=exclude,
    )
    # cluster_id → dominant_in_slots (どの slot index で支配的か)
    dominant_slot_of: dict[int, set[int]] = {}
    for sp_id in dominant:
        dominant_slot_of[sp_id] = set()
    # dominant() は cluster→matched_speaker 形式なので、slot index も取り直す
    # 答弁者 cluster は除外して「議員 (質問者) の dominant」を再計算
    for s_start, s_end, idx in slot_ranges:
        cluster_time: dict[int, float] = {}
        for c in cues:
            sp = c.get("speaker_id")
            if sp is None or sp in exclude:
                continue
            cs = float(c["start"])
            ce = float(c.get("end", cs))
            ov = max(0.0, min(ce, s_end) - max(cs, s_start))
            if ov > 0:
                cluster_time[sp] = cluster_time.get(sp, 0.0) + ov
        if cluster_time:
            top, _ = max(cluster_time.items(), key=lambda x: x[1])
            dominant_slot_of.setdefault(top, set()).add(idx)

    stats = _cluster_speech_stats(cues)

    # 閾値未満で どの slot でも支配的でない cluster を merge 対象に
    to_merge: set[int] = set()
    for sp_id, st in stats.items():
        if dominant_slot_of.get(sp_id):
            continue  # どこかで支配的なら残す
        if st["cue_count"] < min_cues or st["total_duration"] < min_duration:
            to_merge.add(sp_id)

    if not to_merge:
        return cues

    # 各 to_merge cluster の cue を、所属する slot の支配的 cluster に書き換え
    # 所属 slot は cue の start が入る slot
    def _find_slot(t: float) -> int:
        for s_start, s_end, idx in slot_ranges:
            if s_start <= t < s_end:
                return idx
        return -1

    slot_dominant_cluster: dict[int, int] = {}
    for sp_id, slots in dominant_slot_of.items():
        for idx in slots:
            slot_dominant_cluster[idx] = sp_id

    out: list[dict] = []
    for c in cues:
        sp = c.get("speaker_id")
        if sp in to_merge:
            idx = _find_slot(float(c["start"]))
            new_sp = slot_dominant_cluster.get(idx)
            if new_sp is not None:
                c = dict(c)
                c["speaker_id"] = new_sp
                c["_merged_from"] = sp  # debug
        out.append(c)
    return out


def _enrich_clusters_with_text_answerer(
    new_speakers: list[dict],
    cues: list[dict],
    official_speakers: list[dict],
) -> list[dict]:
    """テキスト検出した答弁者 (大臣・参考人等) を cluster の name/group に統合。

    kokkai.shugiin.answerer.inject_answerer_turns で得られる答弁者 entry の
    時刻範囲が、cluster の発話時刻範囲と overlap するなら、cluster に
    その役職と名前を割り当てる (音響 + テキストの統合)。
    """
    try:
        from kokkai.shugiin.answerer import inject_answerer_turns
    except ImportError:
        return new_speakers

    augmented = inject_answerer_turns(official_speakers, cues)
    text_answerers = [s for s in augmented if s.get("auto_detected")]
    if not text_answerers:
        return new_speakers

    stats = _cluster_speech_stats(cues)
    sorted_text = sorted(text_answerers, key=lambda s: s["start"])

    out: list[dict] = []
    for sp in new_speakers:
        sp_id = sp.get("cluster_id")
        if sp_id is None or sp.get("group") in (
            "(答弁者/委員長候補)",
        ):
            # 候補ラベルのみ enrich (議員確定済みのものは触らない)
            pass
        else:
            out.append(sp)
            continue

        st = stats.get(sp_id)
        if not st:
            out.append(sp)
            continue
        c_first = st["first_start"]
        c_last = st["last_end"]

        # cluster 時間範囲内に出現する答弁者を集める
        candidates = [
            t for t in sorted_text
            if c_first <= t["start"] <= c_last
        ]
        if candidates:
            # 最も早い answerer を採用
            t = candidates[0]
            sp = dict(sp)
            sp["name"] = t["name"]
            sp["group"] = t["group"]
            sp["text_enriched"] = True
        out.append(sp)
    return out


def _build_cluster_speakers(
    cues: list[dict],
    official_speakers: list[dict],
    *,
    min_secondary_cues: int = 3,
    min_secondary_duration: float = 10.0,
    use_text_answerer: bool = True,
) -> tuple[list[dict], list[dict]]:
    """cluster ごとに speaker entry を作る + cue 列を整理して返す。

    Args:
        cues: 各 cue が speaker_id を持つ (asr_poc_diarize.py の出力)。
        official_speakers: meta["speakers"] (議員のみ)。
        min_secondary_cues: 非支配 cluster をそのまま残す cue 数の下限。
            これ未満の cluster は所属議員枠の支配 cluster に統合 (誤検出救済)。
        min_secondary_duration: 非支配 cluster をそのまま残す累積秒数の下限。
        use_text_answerer: True なら kokkai.shugiin.answerer.inject_answerer_turns
            の検出結果で cluster の name/role を補完する。

    マッピング規則:
      1. 議員枠内の支配的 cluster = 議員本人
      2. 3 議員枠以上に出現する cluster = "(答弁者/委員長候補)"
      3. 非支配で発話量が閾値未満 → 所属議員枠の支配 cluster に統合 (本人扱い)
      4. 非支配で発話量が十分 → "(○○ 枠内答弁者)"
      5. use_text_answerer=True なら 候補ラベルの cluster をテキスト answerer
         (大臣・参考人) で上書き
    """
    sorted_cues = sorted(cues, key=lambda c: c.get("start", 0.0))
    if not official_speakers:
        return [], sorted_cues

    # 0. 先に「共通答弁者」cluster (大臣・参考人) を特定。これがないと
    #    答弁者 cluster が議員枠の primary を奪い、議員自身の cluster が
    #    「○○ 枠内答弁者」に化ける。
    answerer_clusters = _identify_common_answerer_clusters(
        sorted_cues, official_speakers, min_slots=3,
    )

    # 1. 小さい非支配 cluster を「答弁者を除く」slot dominant に統合
    refined_cues = _merge_tiny_clusters_into_slot_dominant(
        sorted_cues, official_speakers,
        min_cues=min_secondary_cues,
        min_duration=min_secondary_duration,
        exclude_clusters=answerer_clusters,
    )

    # 2. 議員枠の primary は答弁者 cluster を除外して再計算
    dominant = _identify_dominant_cluster_per_member_slot(
        refined_cues, official_speakers, exclude_clusters=answerer_clusters,
    )
    appearance = _cluster_appearance_slots(refined_cues, official_speakers)
    sorted_sp = sorted(official_speakers, key=lambda s: s["start"])

    new_speakers: list[dict] = []
    for c in refined_cues:
        sp_id = c.get("speaker_id")
        if sp_id is None:
            continue
        if any(ns.get("cluster_id") == sp_id for ns in new_speakers):
            continue

        slots_seen = appearance.get(sp_id, set())
        if sp_id in dominant:
            official = dominant[sp_id]["matched_speaker"]
            name = official["name"]
            group = official.get("group", "")
        elif len(slots_seen) >= 3:
            name = f"speaker_{sp_id:02d}"
            group = "(答弁者/委員長候補)"
        elif slots_seen:
            idx = min(slots_seen)
            host = sorted_sp[idx]
            name = f"speaker_{sp_id:02d}"
            group = f"({host['name']} 枠内答弁者)"
        else:
            name = f"speaker_{sp_id:02d}"
            group = "音響クラスタ"

        new_speakers.append({
            "start": float(c["start"]),
            "name": name,
            "group": group,
            "cluster_id": sp_id,
            "auto_detected": True,
        })

    # 2. テキスト answerer 検出結果で「候補」cluster を補完
    if use_text_answerer:
        new_speakers = _enrich_clusters_with_text_answerer(
            new_speakers, refined_cues, official_speakers,
        )

    return sorted(new_speakers, key=lambda s: s["start"]), refined_cues


def main() -> int:
    ap = argparse.ArgumentParser(
        description="話者分離 POC 結果から HTML 生成 (視覚確認用)",
    )
    ap.add_argument(
        "cues_with_speaker", type=Path,
        help="asr_poc_diarize.py の出力 (_cues_with_speaker.json)",
    )
    ap.add_argument(
        "--meta", type=Path, required=True,
        help="公式 meta.json (speakers の時刻照合に使う)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="出力 HTML パス (省略時は入力と同じディレクトリ)",
    )
    ap.add_argument(
        "--min-secondary-cues", type=int, default=3,
        help="議員枠の非支配 cluster をそのまま残す cue 数の下限 (これ未満は"
             " 支配 cluster に統合)",
    )
    ap.add_argument(
        "--min-secondary-duration", type=float, default=10.0,
        help="議員枠の非支配 cluster をそのまま残す累積秒数の下限",
    )
    ap.add_argument(
        "--no-text-answerer", action="store_true",
        help="テキスト answerer 検出 (inject_answerer_turns) で cluster を"
             " 補完しない (音響のみ)",
    )
    args = ap.parse_args()

    # repo root を sys.path に
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kokkai.shugiin.asr import (
        assign_cues_to_speakers,
        merge_cues_into_sentences,
    )
    from kokkai.shugiin.render import render_asr_html

    data = json.loads(args.cues_with_speaker.read_text(encoding="utf-8"))
    cues = data["cues"]
    meta = json.loads(args.meta.read_text(encoding="utf-8"))

    print(f"[poc-render] cue 数: {len(cues)}", file=sys.stderr)
    print(f"[poc-render] 公式 speakers: {len(meta['speakers'])}名", file=sys.stderr)

    new_speakers, sorted_cues = _build_cluster_speakers(
        cues, meta["speakers"],
        min_secondary_cues=args.min_secondary_cues,
        min_secondary_duration=args.min_secondary_duration,
        use_text_answerer=not args.no_text_answerer,
    )

    # 各 cluster の議員マッチ結果を表示
    print(f"\n--- cluster → 議員マッチ結果 (上位 20) ---", file=sys.stderr)
    cluster_counts = defaultdict(int)
    for c in sorted_cues:
        sp = c.get("speaker_id")
        if sp is not None:
            cluster_counts[sp] += 1
    sp_by_cluster = {s["cluster_id"]: s for s in new_speakers}
    for sp_id, n_cue in sorted(cluster_counts.items(), key=lambda x: -x[1])[:20]:
        sp = sp_by_cluster.get(sp_id, {})
        name = sp.get("name", f"speaker_{sp_id:02d}")
        group = sp.get("group", "?")
        print(f"  speaker_{sp_id:02d} cue={n_cue:>3} → {name} ({group})",
              file=sys.stderr)

    print(f"\n[poc-render] 新 speakers リスト: {len(new_speakers)} entry",
          file=sys.stderr)

    # 既存パイプラインで group 化 + 文単位マージ
    groups = assign_cues_to_speakers(sorted_cues, new_speakers)
    total_after = 0
    for g in groups:
        g["cues"] = merge_cues_into_sentences(g["cues"])
        total_after += len(g["cues"])
    print(
        f"[poc-render] cue {len(cues)}件 → 振り分け後 文単位マージで {total_after}件",
        file=sys.stderr,
    )

    # meta は upstream の deli_id / title 等が必要。最小限で構築。
    render_meta = {
        "deli_id": meta.get("id") or meta.get("deli_id", ""),
        "title": meta.get("title", ""),
        "date": meta.get("date", ""),
        "page_url": meta.get("page_url", ""),
        "speakers": new_speakers,
        "agenda": meta.get("agenda", []),
    }

    pipeline_stats = {
        "phase": "phase3-diarize-poc",
        "asr_backend": "(prev)",
        "asr_model": "(prev)",
        "diarization": True,
        "diarize_clusters": len(set(c.get("speaker_id") for c in cues if c.get("speaker_id") is not None)),
        "diarize_matched_speakers": sum(
            1 for s in new_speakers if s.get("group") != "音響クラスタ"
        ),
    }
    html = render_asr_html(render_meta, groups, pipeline=pipeline_stats)

    out_path = args.out
    if out_path is None:
        out_path = args.cues_with_speaker.parent / (
            args.cues_with_speaker.stem.replace("_cues_with_speaker", "") + "_diarize.html"
        )
    out_path.write_text(html, encoding="utf-8")
    print(f"[poc-render] 保存: {out_path}", file=sys.stderr)

    # cluster_map.json: debug / inspection 用に cluster → name/role を保存
    map_path = out_path.with_name(out_path.stem.replace("_diarize", "") + "_cluster_map.json")
    map_path.write_text(
        json.dumps(
            {
                "params": {
                    "min_secondary_cues": args.min_secondary_cues,
                    "min_secondary_duration": args.min_secondary_duration,
                    "use_text_answerer": not args.no_text_answerer,
                },
                "speakers": [
                    {
                        "cluster_id": s.get("cluster_id"),
                        "start": s.get("start"),
                        "name": s.get("name"),
                        "group": s.get("group"),
                        "text_enriched": s.get("text_enriched", False),
                    }
                    for s in new_speakers
                ],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[poc-render] 保存: {map_path}", file=sys.stderr)

    # 分類統計の表示
    n_primary = sum(1 for s in new_speakers
                    if s.get("group") not in ("(答弁者/委員長候補)", "音響クラスタ")
                    and "枠内答弁者" not in (s.get("group") or ""))
    n_candidate = sum(1 for s in new_speakers if s.get("group") == "(答弁者/委員長候補)")
    n_in_slot = sum(1 for s in new_speakers if "枠内答弁者" in (s.get("group") or ""))
    n_text_enriched = sum(1 for s in new_speakers if s.get("text_enriched"))
    print(
        f"\n[poc-render] 分類: 議員確定 {n_primary} / 候補 {n_candidate} / "
        f"枠内答弁者 {n_in_slot} / テキスト統合 {n_text_enriched}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
