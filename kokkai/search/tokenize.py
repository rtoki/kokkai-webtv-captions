"""
SudachiPy ベースの日本語トークナイザ (BM25 用)。

既に sangiin の発言者検出で SudachiPy を依存に持っているので追加コスト 0。
Mode A (短単位、最も細かい分割) を使い、内容語 (名詞・動詞・形容詞)
だけを残してストップワード化を兼ねる。
"""

from __future__ import annotations


# Sudachi のトークナイザはスレッドセーフでないので、プロセス単位でキャッシュ
_TOKENIZER = None
# 除外する品詞 (助詞・助動詞・記号類)。
# 「内容語のみ通す」より「機能語のみ落とす」方式の方が安全 (カタカナ語や
# 「サイバー」のような Sudachi が「形状詞」「接尾辞」等に分類するケースを拾える)。
_EXCLUDE_POS = {"助詞", "助動詞", "補助記号", "記号", "空白", "接続詞"}


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from sudachipy import dictionary, tokenizer  # type: ignore[import-not-found]
        tok = dictionary.Dictionary().create()
        _TOKENIZER = (tok, tokenizer.Tokenizer.SplitMode.A)
    return _TOKENIZER


def tokenize(text: str) -> list[str]:
    """テキストを短単位トークン (内容語のみ、表記正規化済) のリストにする。

    例: "成年後見の制度" → ["成年", "後見", "制度"]
    (Sudachi Mode A は接辞・助詞・助動詞を別トークンに切るので、それらは除外)
    """
    if not text:
        return []
    tok, mode = _get_tokenizer()
    out: list[str] = []
    for m in tok.tokenize(text, mode):
        pos = m.part_of_speech()[0]
        if pos in _EXCLUDE_POS:
            continue
        # 表記揺れ吸収のため normalized_form (正規化) を使う
        norm = m.normalized_form()
        if not norm:
            continue
        # 1 文字の漢字以外 (ひらがな・カタカナ・記号 1 字) はノイズになりやすいので落とす
        if len(norm) == 1 and not _is_kanji(norm):
            continue
        out.append(norm)
    return out


def _is_kanji(c: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in c)


def tokenize_query(query: str) -> tuple[list[str], list[str]]:
    """検索クエリをトークン化し、必須トークンとフレーズに分ける。

    入力:
        スペース区切りの語と "..." で囲んだフレーズ
        例: '成年後見 "事理弁識" 制度'
            → terms=["成年", "後見", "制度"], phrases=["事理弁識"]

    Returns:
        (terms, phrases)
        - terms: 各語を tokenize() で短単位に展開した結果のフラットリスト (AND)
        - phrases: " " で囲まれた文字列 (substring 一致でフィルタする)
    """
    import re
    phrases: list[str] = []
    # ダブルクォートで囲まれた部分を抽出
    def _extract_phrase(m: "re.Match[str]") -> str:
        phrases.append(m.group(1))
        return " "
    rest = re.sub(r'"([^"]+)"', _extract_phrase, query)
    terms: list[str] = []
    for raw_term in rest.split():
        terms.extend(tokenize(raw_term))
    return terms, phrases
