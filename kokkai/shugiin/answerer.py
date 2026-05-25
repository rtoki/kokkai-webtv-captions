"""
ASR cue 列から委員長による答弁者アナウンス (大臣 / 副大臣 / 政府参考人 / 提出者 等)
を検出し、speakers リストに新 turn として挿入するモジュール。

背景: 衆議院 webtv の公式メタ (`extract._extract_speakers`) には質問者の議員
しか含まれず、大臣・政府参考人・副大臣などの発言区間が切り出せない。結果、
HTML 上では議員 1 名のセクションに自分の質問と政府側の答弁が混在し、誰の
発言かが判別できない (`assign_cues_to_speakers` は時刻ベースで議員ブロックに
全て吸収する)。

本モジュールは参議院側 (``kokkai.sangiin.detect``) で既に実装されている
SudachiPy + 役職キーワードの 人名+役職 検出ロジックを ASR cue に対して走らせ、
検出された人を新 speaker turn として ``speakers`` に追加する。後段の
``asr.assign_cues_to_speakers`` は時刻ベースなので、turn が追加されれば
HTML 上に自動的に答弁者ごとのセクションが生まれる。

注意:
- 委員長セクション内の cue だけを対象にする (議員質問中の「○○大臣が」言及で
  誤検出するのを防ぐ)。
- ``name`` は ASR が拾った漢字なので誤読の可能性あり (例: 「鳩山次郎大臣」)。
  議員名簿との突合は別タスク。
- ``role_str`` (group) は ``classify_role_str`` で minister / bureaucrat 等に
  分類できる。
"""

from __future__ import annotations

from kokkai.sangiin.detect import classify_role_str, detect_speaker_calls


def inject_answerer_turns(
    speakers: list[dict],
    cues: list[dict],
) -> list[dict]:
    """``speakers`` (時系列) に大臣 / 政府参考人 / 提出者 等の turn を挿入して返す。

    挿入される entry:

        ``{"start": float, "name": str, "group": str, "auto_detected": True}``

    検出対象は **全 cue**。委員長は議員交代時に毎回「では○○大臣」とアナウンス
    するが、衆議院 webtv の公式 speakers リストには委員長セクションが各議員の前に
    入っておらず、議員のセクション内に委員長発話が埋まっている。そのため委員長
    セクション限定にすると検出ほぼゼロになる。

    false positive (議員質問中の「○○大臣は」言及) は ``sangiin.detect`` が
    『役職直後が助詞なら言及として除外』する判定で抑えている。

    Args:
        speakers: ``[{"start", "name", "group"}, ...]`` (公式メタ由来、議員のみ)
        cues: ASR の cue 列 ``[{"start", "end", "text"}, ...]``

    Returns:
        元 ``speakers`` + 検出した turn を時刻昇順にマージした新 list。
        ``cues`` が空・``speakers`` が空の場合は元 list と同等を返す。
    """
    if not speakers or not cues:
        return list(speakers)

    new_entries: list[dict] = []
    # 同じ (時刻, 人名, 役職) の重複を防ぐ
    seen: set[tuple[float, str, str]] = set()

    for cue in cues:
        try:
            calls = detect_speaker_calls(cue["text"])
        except Exception:
            # SudachiPy が落ちても他の cue は走らせ続ける
            continue
        for call in calls:
            role_str = call.get("role_str", "")
            # 質問者復帰 ('_return' = 「○○くん」) は新 turn ではないので除外
            if role_str == "_return":
                continue
            klass = classify_role_str(role_str)
            # 議員自身への言及 (questioner) や役職外 (other) は除外。
            # chair (委員長) も新 turn として追加しない (公式メタに既にあるか、
            # 議員セクション内の繋ぎ発話は分離する価値が低い)。
            if klass in ("questioner", "chair", "other"):
                continue
            name = call.get("name", "").strip()
            if not name:
                continue
            # アナウンス cue の直後を新 turn の start に
            start = float(cue["end"]) + 0.01
            key = (round(start, 1), name, role_str)
            if key in seen:
                continue
            seen.add(key)
            new_entries.append({
                "start": start,
                "name": name,
                "group": role_str,
                "auto_detected": True,
            })

    return sorted(speakers + new_entries, key=lambda s: s["start"])
