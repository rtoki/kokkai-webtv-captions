"""
faster-whisper の ``initial_prompt`` を構築するモジュール。

Whisper モデルへの prompt は約 224 トークン (~150 日本語文字) しか入らないため、
情報を優先順位付けして詰める:

  Tier 1 (必須): 会議名 + 日付
  Tier 2 (高): 発言者氏名 (公式リスト由来、当日確実に話す)
  Tier 3 (中): 発言者所属会派 (組織名固有名詞のバイアス)
  Tier 4 (中): 議題に含まれる固有語 (法案名から抽出)
  Tier 5 (低): 同会派の他議員 (割込発言・他党質疑に備える)
  Tier 6 (低): 議会頻出語 (大臣 / 政府参考人 / 委員長 …)

優先度の低いものは予算オーバーで切り捨てる。

公開関数:
- ``build_initial_prompt(meta, members, max_chars=160)``: 文字列を返す
"""

from __future__ import annotations

import re


COMMON_PARLIAMENT_TERMS = [
    # 役職
    "委員長", "大臣", "副大臣", "政府参考人", "提出者", "理事",
    # 議事進行
    "答弁", "質疑", "質問", "趣旨説明", "採決", "可決", "付託", "聴取",
    # 法律一般 (法務・財金・内閣委員会等で頻出)
    "民法", "改正", "施行", "附則",
    # 民法分野 (成年後見・遺言など Whisper turbo が誤変換しやすい語)
    "後見", "保佐", "補助", "成年後見", "事理弁識", "家庭裁判所",
]

# 法案番号 (例: "(221国会閣30)", "（第217回国会閣75号）") は議題テキストから
# 削除する — ASR bias に不要 + 「国会閣」のようなノイズ語が混入するため。
_BILL_NUMBER_RE = re.compile(r"[（(][^（()）]*\d+[^（()）]*[）)]")

# 機能語塊 (3 文字以上の漢字塊として現れるが固有名詞でないもの)
_AGENDA_NOUN_STOPLIST = {
    "に関する", "等の", "に基づく", "ことによる", "について",
    "に対する", "の一部", "を一部",
}


def _extract_proper_nouns_from_agenda(agenda_items: list[str]) -> list[str]:
    """
    法案名から固有名詞 (組織名・法律名の一部) を粗く抽出する。

    例: "経済施策を一体的に講ずることによる安全保障の確保の推進に関する法律及び
         株式会社国際協力銀行法の一部を改正する法律案（221国会閣30）"
        → ['経済施策', '安全保障', '株式会社国際協力銀行法']

    法案番号 (NNN国会閣NN) は事前に除去する。
    漢字+カタカナの 3 文字以上の塊を拾い、機能語ストップリストで filter。
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in agenda_items:
        item = _BILL_NUMBER_RE.sub("", item)
        for m in re.findall(r"[一-龥ァ-ヶー]{3,}", item):
            if m in seen or m in _AGENDA_NOUN_STOPLIST:
                continue
            seen.add(m)
            out.append(m)
    return out


def _faction_colleagues(
    speaker_factions: list[str],
    members: list[dict],
    exclude_names: set[str],
    limit_per_faction: int = 3,
) -> list[str]:
    """発言者と同会派の議員を最大 limit_per_faction 名ずつ拾う。

    答弁差し替え・他党質疑時に名前が出るケースを想定した補完。
    """
    out: list[str] = []
    for faction in speaker_factions:
        n = 0
        for m in members:
            if m.get("faction") != faction:
                continue
            if m["name"] in exclude_names:
                continue
            out.append(m["name"])
            n += 1
            if n >= limit_per_faction:
                break
    return out


def build_initial_prompt(
    meta: dict,
    members: list[dict] | None = None,
    max_chars: int = 160,
) -> str:
    """
    Whisper への initial_prompt 文字列を構築する。

    Args:
        meta: ``kokkai.shugiin.extract.resolve_from_shugiin_detail`` が返す dict
        members: ``kokkai.shugiin.members.load_members()`` の結果。None なら
                 同会派補完を skip する。
        max_chars: 出力上限文字数 (Whisper の prompt token 制約への安全マージン)

    Returns:
        改行入りの一文字列。faster-whisper の ``initial_prompt`` にそのまま渡せる。
    """
    speakers = meta.get("speakers") or []
    # 同じ発言者が複数回登場するケース (途中で委員長が割り込む等) は dedup
    speaker_names: list[str] = []
    seen_names: set[str] = set()
    for s in speakers:
        n = s["name"]
        if n in seen_names:
            continue
        seen_names.add(n)
        speaker_names.append(n)
    # group フィールドが「○○委員長」「○○大臣」のような役職表記の場合は
    # 会派ではないので除外する (本来は政党名・会派名のみ会派欄に出したい)
    role_suffix_re = re.compile(
        r"(?:委員長|議長|副議長|大臣|副大臣|政務官|政府参考人)$"
    )
    speaker_factions: list[str] = []
    seen_fac: set[str] = set()
    for s in speakers:
        f = s.get("group", "").strip()
        if not f:
            continue
        if role_suffix_re.search(f):
            continue
        if f in seen_fac:
            continue
        seen_fac.add(f)
        speaker_factions.append(f)

    parts: list[str] = []

    # Tier 1: 会議名 + 日付
    head = f"{meta.get('title', '')}（{meta.get('date', '')}）。".strip("（）")
    parts.append(head)

    # Tier 2: 発言者氏名
    if speaker_names:
        parts.append(f"発言者: {'、'.join(speaker_names)}。")

    # Tier 3: 会派
    if speaker_factions:
        parts.append(f"会派: {'、'.join(speaker_factions)}。")

    # Tier 4: 議題からの固有名詞
    nouns = _extract_proper_nouns_from_agenda(meta.get("agenda") or [])
    if nouns:
        parts.append(f"議題: {'、'.join(nouns)}。")

    # Tier 5: 同会派補完
    if members:
        colleagues = _faction_colleagues(
            speaker_factions, members, set(speaker_names)
        )
        if colleagues:
            parts.append(f"関連議員: {'、'.join(colleagues)}。")

    # Tier 6: 議会頻出語
    parts.append(f"用語: {'、'.join(COMMON_PARLIAMENT_TERMS)}。")

    # 優先度低い順に削って max_chars に収める
    while parts and sum(len(p) for p in parts) > max_chars:
        parts.pop()  # 末尾 (= 優先度低) から落とす

    return "\n".join(parts)
