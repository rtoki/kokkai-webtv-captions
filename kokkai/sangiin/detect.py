"""
発言者検出 (SudachiPy + 役職キーワード).

参議院 AI 字幕の本文中から「人名 + 役職語」のパターンを検出し、
誰の発言が始まる位置かを抽出する。

中核は ``detect_speaker_calls(text)``。
"""

from __future__ import annotations

from typing import Iterable

# 役職キーワード (後方一致でマッチ・長い表現を先に並べる)
# 参議院・委員会・本会議で実際に登場する役職語を網羅。
ROLE_KEYWORDS: tuple[str, ...] = (
    # === 政治家・大臣系 ===
    "内閣総理大臣", "総理大臣",
    "国務大臣", "副大臣",
    "大臣政務官", "政務官",
    "総理", "首相", "大臣",
    # === 議会役職 ===
    "議長", "副議長",
    "委員長", "副委員長",
    "理事",
    # === 政府参考人系 ===
    "政府参考人", "参考人",
    # === 官房系 ===
    "内閣官房副長官", "内閣官房長官",
    "官房副長官", "官房長官",
    # === 審議官系 ===
    "総括審議官", "審議官",
    "参事官",
    # === 統括系 ===
    "統括監", "統括官",
    # === 局長・長官系 ===
    "推進事務局長", "事務局長",
    "事務総長",
    "局長", "次長", "長官", "室長",
    # === 自衛隊・防衛 ===
    "統合幕僚長",
    "陸上幕僚長", "海上幕僚長", "航空幕僚長",
    # === 司法 ===
    "検事総長",
    "最高裁長官",
)

# 「○○くん/君/氏」 = 質問者復帰の呼び戻し
RETURN_SUFFIXES: tuple[str, ...] = ("くん", "君", "氏")

# 政府参考人・官僚系 (描画分類用)
BUREAUCRAT_KEYS: tuple[str, ...] = (
    "参考人", "統括監", "統括官", "審議官", "参事官",
    "局長", "次長", "長官", "室長",
    "官房長", "事務局", "事務総長",
    "幕僚長", "検事", "最高裁",
)
# 大臣系
MINISTER_KEYS: tuple[str, ...] = ("大臣", "副大臣", "政務官", "総理", "首相")
# 議会役職 (questioner ではなく chair 系統)
CHAIR_KEYS: tuple[str, ...] = ("委員長", "副委員長", "議長", "副議長", "理事")

# 形態素的に人名候補とみなす POS (3層まで一致)
PERSON_POS: set[tuple[str, str, str]] = {
    ("名詞", "固有名詞", "人名"),
    ("名詞", "固有名詞", "地名"),  # 松本/榊原/坂越 等は地名扱いになる
}

# 固有名詞 POS でも除外したい (照応・著名地名)
NAME_STOPLIST: set[str] = {
    "今", "先", "前", "次", "別", "他", "今後", "今日", "今回", "今般",
    "本日", "本年", "明日", "昨日",
    "東京", "日本", "国会", "参議院", "衆議院",
    "我が国", "我国", "我",
}

# 助詞 POS (Sudachi では大分類「助詞」)
PARTICLE_POS = "助詞"


_SUDACHI_TOK = None


def _sudachi():
    global _SUDACHI_TOK
    if _SUDACHI_TOK is None:
        from sudachipy import Dictionary
        _SUDACHI_TOK = Dictionary().create()
    return _SUDACHI_TOK


_PUNCT_RE = "、。,.、。 "


def _find_role_suffix(surfaces: Iterable[str]) -> str | None:
    """連続する surface 列を joined して、後方一致で役職語にマッチさせる。
    句読点が含まれた surface 列は役職フレーズの境界とみなし無効とする。
    """
    joined = "".join(surfaces)
    if any(p in joined for p in _PUNCT_RE):
        return None
    for kw in ROLE_KEYWORDS:
        if joined.endswith(kw):
            return joined
    return None


# SudachiPy の 1 回 tokenize() 入力上限は 49,149 byte。
# 長尺会議 (本会議や予算委員会で 50KB+) はこの上限を踏むので、文末で安全に
# 区切ってチャンク単位で tokenize → 元 text 上の offset に補正してマージする。
# 安全マージンを取って 40KB を 1 チャンクの上限に。
_SUDACHI_MAX_BYTES = 40_000


def _chunk_by_sentence(text: str, max_bytes: int = _SUDACHI_MAX_BYTES) -> list[tuple[int, str]]:
    """text を 「。」「！」「？」「\n」で安全に分割。

    Returns:
        ``[(start_char_offset, chunk_text), ...]`` — start_char_offset は元 text 内の文字位置
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return [(0, text)]

    boundaries = {"。", "！", "？", "\n"}
    chunks: list[tuple[int, str]] = []
    cur_start = 0
    cur = []
    cur_bytes = 0
    last_boundary = -1   # cur 内での最後の境界 (cur index)
    for ch in text:
        cur.append(ch)
        cur_bytes += len(ch.encode("utf-8"))
        if ch in boundaries:
            last_boundary = len(cur)
        if cur_bytes >= max_bytes:
            cut = last_boundary if last_boundary > 0 else len(cur)
            piece = "".join(cur[:cut])
            chunks.append((cur_start, piece))
            cur_start += len(piece)
            cur = cur[cut:]
            cur_bytes = sum(len(c.encode("utf-8")) for c in cur)
            last_boundary = -1
    if cur:
        chunks.append((cur_start, "".join(cur)))
    return chunks


def detect_speaker_calls(text: str) -> list[dict]:
    """
    本文を SudachiPy でトークナイズし、「人名/地名 + 役職語(複合可)」のパターンを
    全て検出する。直後が助詞 (の/に/は/が…) の場合は『言及』として除外。

    返値: ``[{"name", "role_str", "start", "end", "person_end_offset"}, ...]``

    role_str の特殊値:
    - ``"_return"``: 「○○くん/君/氏」 (質問者復帰のコール)

    長尺 text (SudachiPy の 49,149 byte 上限超) は ``_chunk_by_sentence`` で
    自動分割し、各チャンクの結果を元 text 上の offset に補正してマージする。
    """
    chunks = _chunk_by_sentence(text)
    all_calls: list[dict] = []
    for chunk_start_char, chunk_text in chunks:
        all_calls.extend(_detect_in_chunk(chunk_text, chunk_start_char))
    return all_calls


def _detect_in_chunk(text: str, base_offset: int) -> list[dict]:
    from sudachipy import tokenizer as st
    tok = _sudachi()
    morphs = list(tok.tokenize(text, st.Tokenizer.SplitMode.C))

    # 各形態素の text 上 offset
    offsets = [0]
    for m in morphs:
        offsets.append(offsets[-1] + len(m.surface()))

    calls: list[dict] = []
    i = 0
    while i < len(morphs):
        pos3 = tuple(morphs[i].part_of_speech()[:3])
        if pos3 not in PERSON_POS:
            i += 1
            continue
        # 連続する固有名詞をまとめる (姓+名)
        person_start = i
        person_end = i + 1
        while (
            person_end < len(morphs)
            and tuple(morphs[person_end].part_of_speech()[:3]) in PERSON_POS
        ):
            person_end += 1
        person_name = "".join(morphs[k].surface() for k in range(person_start, person_end))
        if person_name in NAME_STOPLIST:
            i = person_end
            continue

        # 直後 1〜4 トークンに役職語があるか
        matched: tuple[int, str] | None = None
        for span in range(1, 5):
            end = person_end + span
            if end > len(morphs):
                break
            surfaces = [morphs[k].surface() for k in range(person_end, end)]
            role_str = _find_role_suffix(surfaces)
            if role_str:
                matched = (end, role_str)
                break
            joined = "".join(surfaces)
            if joined in RETURN_SUFFIXES:
                matched = (end, "_return")
                break

        if matched:
            role_end, role_str = matched
            # 役職の直後が助詞 (の/に/は/が…) なら『言及』として除外
            after_pos = morphs[role_end].part_of_speech()[0] if role_end < len(morphs) else ""
            if after_pos == PARTICLE_POS:
                i = person_end
                continue
            calls.append({
                "name": person_name,
                "role_str": role_str,
                "start": base_offset + offsets[person_start],
                "end": base_offset + offsets[role_end],
                "person_end_offset": base_offset + offsets[person_end],
            })
            i = role_end
        else:
            i = person_end
    return calls


def classify_role_str(role_str: str) -> str:
    """役職文字列を questioner / chair / minister / bureaucrat / other に分類。"""
    if role_str == "_return":
        return "questioner"
    if any(k in role_str for k in CHAIR_KEYS):
        return "chair"
    if any(k in role_str for k in BUREAUCRAT_KEYS):
        return "bureaucrat"
    if any(k in role_str for k in MINISTER_KEYS):
        return "minister"
    return "other"


def detect_turns(
    section_text: str, primary_name: str, primary_group: str = ""
) -> list[dict]:
    """
    議員1名分のセクションを発言者ターンに切り分ける。

    検出された各 'call' の start offset を切れ目とする。
    role='_return' (○○くん) は質問者復帰として扱う。
    """
    text = section_text.strip()
    if not text:
        return []

    calls = detect_speaker_calls(text)
    if not calls:
        return [{
            "speaker": primary_name, "group": primary_group,
            "role": "議員", "klass": "questioner",
            "text": text,
        }]

    segments: list[dict] = []
    # 先頭から最初の call までは questioner
    first_start = calls[0]["start"]
    if first_start > 0:
        head = text[:first_start].strip()
        if head:
            segments.append({
                "speaker": primary_name, "group": primary_group,
                "role": "議員", "klass": "questioner", "text": head,
            })

    for i, c in enumerate(calls):
        seg_start = c["start"]
        seg_end = calls[i + 1]["start"] if i + 1 < len(calls) else len(text)
        seg_text = text[seg_start:seg_end].strip()
        if not seg_text:
            continue
        if c["role_str"] == "_return":
            speaker = primary_name
            klass = "questioner"
            role_disp = "議員"
        else:
            speaker = c["name"]
            klass = classify_role_str(c["role_str"])
            role_disp = c["role_str"]
            if c["name"] == primary_name:
                klass = "questioner"
                role_disp = "議員"
        segments.append({
            "speaker": speaker, "group": "",
            "role": role_disp, "klass": klass, "text": seg_text,
        })

    # 連続する同一 speaker+klass セグメントを統合
    merged: list[dict] = []
    for s in segments:
        if (
            merged
            and merged[-1]["speaker"] == s["speaker"]
            and merged[-1]["klass"] == s["klass"]
        ):
            merged[-1]["text"] += " " + s["text"]
        else:
            merged.append(s)
    return merged


def group_cues_by_speaker(cues: list[dict], speakers: list[dict]) -> list[dict]:
    """字幕 cue を発言者リスト (時系列) で区切ってグループ化。"""
    if not speakers:
        return [{
            "speaker": {"name": "全体", "group": "", "start": 0.0},
            "text": " ".join(c["text"] for c in cues),
        }]
    results = []
    for i, sp in enumerate(speakers):
        end = speakers[i + 1]["start"] if i + 1 < len(speakers) else float("inf")
        text = " ".join(c["text"] for c in cues if sp["start"] <= c["start"] < end)
        results.append({"speaker": sp, "text": text})
    return results
