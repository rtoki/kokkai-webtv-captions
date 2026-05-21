"""
ASR 出力の静的置換 (誤 → 正) を管理するモジュール。

Whisper turbo は議会音声の固有名詞・専門用語を頻繁に音声的に近い誤字に変換する
(例: 「全会一致」→「前回一致」、「附帯決議」→「不対決議」、「対内直接投資」→
「体内直接投資」)。LLM 校正 (``llm_correct.py``) で context-aware 修正をかける前に、
**確実に誤りと判っているパターンは事前に置換**しておくことで:

- 高速 (regex 数十回の置換)
- 決定論的 (LLM 揺らぎなし)
- LLM 校正で扱う token 数を減らす (コスト・レイテンシ削減)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# 議会音声で頻繁に観測される Whisper turbo の誤変換パターン。
# 法務委員会・本会議など複数の会議での実走で確認した誤りを中心に収録。
#
# 「誤 → 正」形式。短い方を先に置換すると誤って長い正解を壊すので、
# 長いキーから順に適用する (apply 時にソート)。
DEFAULT_PARLIAMENT_GLOSSARY: dict[str, str] = {
    # 議事手続き系 (本会議で頻出)
    "前回一致": "全会一致",
    "不対決議": "附帯決議",
    "規律多数": "起立多数",
    "規律を求めます": "起立を求めます",
    "起立を求めます。規律": "起立を求めます。起立",
    "委員長報酷": "委員長報告",
    "報酷": "報告",
    "时代となりました": "議題となりました",
    "时代となります": "議題となります",
    "一定第": "日程第",
    "の报告": "の報告",

    # 法律用語 (法務委員会で頻出)
    "青年貢献": "成年後見",
    "自理を弁識": "事理を弁識",
    "公権及び補佐": "後見及び保佐",
    "公権開始": "後見開始",
    "補佐開始": "保佐開始",
    "補助開始の審刊": "補助開始の審判",
    "の審刊": "の審判",
    "応印要件": "押印要件",
    "に閣する": "に関する",
    "任意貢献契約": "任意後見契約",
    "任意貢権監督人": "任意後見監督人",
    "任意貢監督人": "任意後見監督人",
    "貢献登記": "後見登記",
    "貢献開始": "後見開始",
    "法廷の重要な財産": "法定の重要な財産",
    "御異義なし": "御異議なし",

    # 経済・財政系
    "体内直接投資": "対内直接投資",
    "体内直接投信": "対内直接投資",
    "财务金優": "財務金融",
    "财务金融": "財務金融",
    "財務金優": "財務金融",

    # 通信・規制系
    "携帰音声通信": "携帯音声通信",
    "携帰": "携帯",
    "電気通信駅務": "電気通信役務",
    "電気通信駆務": "電気通信役務",
    "電気通信益務": "電気通信役務",
    "電気通信駅": "電気通信役",
    "音声通信駅務": "音声通信役務",
    "益務を提供": "役務を提供",

    # その他
    "新政英": "生成 AI",  # 文脈次第で誤る可能性ある保留候補だが多くの文脈で正
}


def load_glossary(path: Path | None, include_defaults: bool = True) -> dict[str, str]:
    """
    glossary ファイルを読んで dict[誤, 正] を返す。

    ファイルフォーマット (UTF-8, 1 行 1 エントリ):
        # コメント行
        誤 → 正
        誤 -> 正

    Args:
        path: 追加 glossary ファイル (None なら追加なし)
        include_defaults: True なら ``DEFAULT_PARLIAMENT_GLOSSARY`` も含める

    Returns:
        dict[誤, 正]。ユーザー定義が default を上書き。
    """
    merged: dict[str, str] = {}
    if include_defaults:
        merged.update(DEFAULT_PARLIAMENT_GLOSSARY)

    if path is None or not path.exists():
        return merged

    arrow = re.compile(r"\s*(?:→|->)\s*")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = arrow.split(line, maxsplit=1)
        if len(parts) != 2:
            continue
        wrong, right = parts[0].strip(), parts[1].strip()
        if wrong and right:
            merged[wrong] = right
    return merged


def apply_glossary(cues: list[dict], glossary: dict[str, str]) -> int:
    """
    cue リストに静的置換を適用 (破壊的)。

    長いキーから順に置換することで、短いキーが長いキーの一部を破壊する事故を防ぐ
    (例: 「補助開始の審刊」を先に修正してから「審刊」を修正)。

    Returns:
        修正件数 (cue 単位)
    """
    if not glossary:
        return 0
    pairs = sorted(glossary.items(), key=lambda kv: -len(kv[0]))

    count = 0
    for c in cues:
        text = c.get("text", "")
        new = text
        for wrong, right in pairs:
            if wrong in new:
                new = new.replace(wrong, right)
        if new != text:
            c["text"] = new
            count += 1
    return count


def report(stats_count: int, glossary_size: int) -> None:
    """ログ出力ヘルパ"""
    print(
        f"[glossary] 静的置換適用: {stats_count} cue 修正 (辞書 {glossary_size} 項目)",
        file=sys.stderr,
    )
