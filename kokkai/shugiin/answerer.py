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

from collections import defaultdict

from kokkai.sangiin.detect import classify_role_str, detect_speaker_calls


# chair アナウンス cue の前置接続詞 (「では、〇〇大臣」のように冒頭に来やすい)。
# 長 cue でこの prefix の後に call が来ているなら委員長アナウンスと判定する。
_CHAIR_INTRO_PREFIX_CHARS = 12

# 委員長アナウンスとみなす最大 cue 長 (短い cue = bare announcement)。
_CHAIR_INTRO_SHORT_CUE_CHARS = 40


def _looks_like_chair_intro(cue_text: str, call_offset_in_cue: int) -> bool:
    """``cue_text`` 中の call が委員長アナウンスっぽいかの heuristic 判定。

    True を返すのは:
    - cue 全体が ``_CHAIR_INTRO_SHORT_CUE_CHARS`` 文字未満 (bare な「○○大臣」)
    - もしくは call が cue 冒頭 ``_CHAIR_INTRO_PREFIX_CHARS`` 文字以内
      (「では、○○大臣」「次に、○○大臣」許容)

    False になるケース: 議員質問中の「私から○○大臣にお伺いします」のように
    call が長い文の中盤にあるパターン。
    """
    text = (cue_text or "").strip()
    if len(text) < _CHAIR_INTRO_SHORT_CUE_CHARS:
        return True
    return call_offset_in_cue <= _CHAIR_INTRO_PREFIX_CHARS


def _find_original_questioner_by_surname(
    sorted_originals: list[dict],
    surname: str,
    before_time: float,
) -> dict | None:
    """``sorted_originals`` の中から、``before_time`` 以前に登壇した同姓の公式
    speaker を最も新しいもの優先で 1 件返す (見つからなければ None)。

    surname は姓のみ (1〜3 文字) を想定。完全マッチ → 前方一致 の順で探す。
    """
    if not surname:
        return None
    candidates = [
        s for s in sorted_originals
        if s.get("start", 0.0) <= before_time
        and ((s.get("name") or "") == surname
             or (s.get("name") or "").startswith(surname))
    ]
    if not candidates:
        return None
    return candidates[-1]


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

    false positive (議員質問中の「○○大臣にお伺いします」) は 2 段で抑える:

    1. ``sangiin.detect`` が『役職直後が助詞なら言及として除外』
    2. 本関数の ``_looks_like_chair_intro`` heuristic (cue 長 + call 位置)

    `_return` ("○○くん") を検出した場合は、直前の auto_detected (= 大臣等)
    turn を打ち切るために、公式 speakers 中の同姓質問者に戻る turn を挿入する。
    これがないと「大臣答弁の後、再質問に戻ったのに大臣セクションが続く」誤帰属が
    残る。

    Args:
        speakers: ``[{"start", "name", "group"}, ...]`` (公式メタ由来、議員のみ)
        cues: ASR の cue 列 ``[{"start", "end", "text"}, ...]``

    Returns:
        元 ``speakers`` + 検出した turn を時刻昇順にマージした新 list。
        ``cues`` が空・``speakers`` が空の場合は元 list と同等を返す。
    """
    if not speakers or not cues:
        return list(speakers)

    # 公式 speakers (auto_detected フラグなし) を time 順に並べる。
    # `_return` のときの「直前の同姓質問者」検索に使う。
    sorted_originals = sorted(
        (s for s in speakers if not s.get("auto_detected")),
        key=lambda s: float(s.get("start", 0.0)),
    )

    new_entries: list[dict] = []
    seen: set[tuple[float, str, str]] = set()

    # 直近に挿入した turn が大臣・政府参考人系なら True。
    # `_return` を使って「議員へ戻る」 turn を入れるかの判定に使う。
    last_injected_is_answerer = False

    for cue in cues:
        try:
            calls = detect_speaker_calls(cue["text"])
        except Exception:
            # SudachiPy が落ちても他の cue は走らせ続ける
            continue
        cue_text = cue.get("text") or ""
        # cue 内の call.start は元 text 上の offset。base_offset の関係で
        # cue 全文の先頭からの相対位置として使えるはず (cue.text 単位で
        # detect_speaker_calls を呼んでいるので)。
        for call in calls:
            role_str = call.get("role_str", "")
            name = (call.get("name") or "").strip()
            call_offset = int(call.get("start", 0))

            # 質問者復帰 (`_return` = 「○○くん」) は委員長が質問者を再度呼び出し
            # たマーカ。直前 auto-detected (大臣) turn があるときに限り、公式
            # speakers 中の同姓質問者に戻る turn を挿入する。
            if role_str == "_return":
                if not last_injected_is_answerer:
                    continue
                # cue.end ちょうどを使う (`+ 0.01` を足すと、次の cue が同時刻
                # から始まる場合に 1 cue 分前のセクションへ取り込まれる)
                cue_end = float(cue["end"])
                orig = _find_original_questioner_by_surname(
                    sorted_originals, name, before_time=cue_end,
                )
                if orig is None:
                    continue
                key = (round(cue_end, 1), orig.get("name", ""), "_return")
                if key in seen:
                    continue
                seen.add(key)
                new_entries.append({
                    "start": cue_end,
                    "name": orig.get("name", ""),
                    "group": orig.get("group", ""),
                    "auto_detected": True,
                    "return_to_questioner": True,
                })
                last_injected_is_answerer = False
                continue

            klass = classify_role_str(role_str)
            # 議員自身への言及 (questioner) や役職外 (other) は除外。
            # chair (委員長) も新 turn として追加しない (公式メタに既にあるか、
            # 議員セクション内の繋ぎ発話は分離する価値が低い)。
            if klass in ("questioner", "chair", "other"):
                continue
            if not name:
                continue
            # heuristic: 質問者が「○○大臣にお伺い」のように長文の中で言及する
            # ケースを弾く。委員長アナウンスは bare ("○○大臣") か短い前置詞
            # 付き ("では○○大臣") のはず。
            if not _looks_like_chair_intro(cue_text, call_offset):
                continue
            # アナウンス cue の終端を新 turn の start に。`+ 0.01` を足すと、
            # 次の cue (= 答弁本文) が同時刻から始まる場合に 1 cue 分が前の
            # 質問者セクションに吸われる (assign_cues_to_speakers が >= で
            # 振り分けるため)。cue.end ちょうどに合わせる。
            start = float(cue["end"])
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
            last_injected_is_answerer = True

    return sorted(speakers + new_entries, key=lambda s: s["start"])


def normalize_answerer_names(
    speakers: list[dict],
    members: list[dict],
    *,
    only_auto_detected: bool = True,
) -> tuple[list[dict], int]:
    """``inject_answerer_turns`` で検出した答弁者 (大臣・副大臣・政務官・提出者
    等の議員系) の name を、衆議院議員名簿で **フルネームに正規化** する。

    背景: ASR が答弁者アナウンスを拾うとき、姓のみ (「山田」「伊藤」) しか
    取れないことが多い。HTML 上で「山田 (副大臣)」「伊藤 (大臣)」のように姓
    だけが表示されていると、誰なのか同定できない。``members.load_members()``
    の議員名簿で姓ユニーク一致するなら、フルネームに置換する。

    マッチ規則 (保守的):

    1. ``name`` が議員名簿の ``name`` と完全一致 → 既に canonical、触らない。
    2. ``name`` が 1-2 文字 (姓と推定) で議員名簿に同じ姓を持つ議員が
       **1 名のみ** → そのフルネームに置換。
    3. それ以外 (複数候補 / マッチ無し / 3 文字以上の不完全名) → そのまま。

    候補が複数のとき (山田姓は複数の議員が居る等) は曖昧なので false positive
    回避のために置換しない。

    Args:
        speakers: ``inject_answerer_turns`` 後の speakers リスト。
        members: ``members.load_members()`` の返り値 (議員 dict のリスト)。
        only_auto_detected: True なら ``auto_detected: True`` の entry のみ対象。
            既存の公式メタ由来の議員 (元の質問者) は触らない。

    Returns:
        ``(新 speakers list, 正規化された entry 数)``。元の name は ``original_name``
        フィールドに退避する。
    """
    if not speakers or not members:
        return list(speakers), 0

    # 議員名簿から (フルネーム集合, 姓→候補リスト) のインデックスを作る。
    # 姓は name 先頭 1 文字 / 2 文字 の両方で引けるようにしておく。
    by_full_name: set[str] = set()
    by_surname_1: dict[str, list[str]] = defaultdict(list)
    by_surname_2: dict[str, list[str]] = defaultdict(list)
    for m in members:
        full = (m.get("name") or "").strip()
        if not full:
            continue
        by_full_name.add(full)
        by_surname_1[full[:1]].append(full)
        if len(full) >= 2:
            by_surname_2[full[:2]].append(full)

    out: list[dict] = []
    n_normalized = 0
    for s in speakers:
        if only_auto_detected and not s.get("auto_detected"):
            out.append(s)
            continue
        nm = (s.get("name") or "").strip()
        if not nm:
            out.append(s)
            continue
        # 既にフルネームなら触らない
        if nm in by_full_name:
            out.append(s)
            continue
        # 姓 (1-2 文字) 一致の候補を集める。長い方 (2 文字) を優先。
        candidates: list[str] = []
        if len(nm) == 2 and nm in by_surname_2:
            candidates = by_surname_2[nm]
        elif len(nm) == 1 and nm in by_surname_1:
            candidates = by_surname_1[nm]
        # 一意に決まるときだけ置換 (複数候補は曖昧なのでスキップ)
        if len(candidates) == 1:
            new = dict(s)
            new["original_name"] = nm
            new["name"] = candidates[0]
            out.append(new)
            n_normalized += 1
        else:
            out.append(s)
    return out, n_normalized
