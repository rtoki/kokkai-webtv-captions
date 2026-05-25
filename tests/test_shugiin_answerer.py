"""answerer.inject_answerer_turns の純関数テスト。

SudachiPy は依存だが既に core 依存に入っており、人名+役職検出は
sangiin.detect に既存実装あり (テスト済)。ここでは shugiin 側の
- 委員長セクション限定の絞り込み
- speakers リストへの新 turn 挿入
- 重複排除
の挙動を仮名 (架空の姓・架空の役職) で検証する。
"""

from __future__ import annotations

import pytest

# SudachiPy が無い環境では importorskip
pytest.importorskip("sudachipy")

from kokkai.shugiin.answerer import inject_answerer_turns


def test_minister_mention_with_particle_is_skipped():
    """『○○大臣にお伺い』のように直後に助詞が続くケースは『言及』として除外
    される (sangiin.detect の既存判定)。これにより議員質問中の言及で誤検出しない。"""
    speakers = [{"start": 100.0, "name": "乙田", "group": "自由民主党"}]
    cues = [{"start": 110.0, "end": 112.0, "text": "甲野大臣にお伺いします"}]
    out = inject_answerer_turns(speakers, cues)
    # 『甲野大臣に』(に=助詞) は言及扱いで挿入されない
    auto = [s for s in out if s.get("auto_detected")]
    assert len(auto) == 0


def test_empty_inputs():
    """speakers / cues が空なら元 list を返す。"""
    assert inject_answerer_turns([], []) == []
    assert inject_answerer_turns([{"start": 0, "name": "x", "group": ""}], []) == [
        {"start": 0, "name": "x", "group": ""}
    ]
    assert inject_answerer_turns([], [{"start": 0, "end": 1, "text": "甲野大臣"}]) == []


def test_chair_announces_minister():
    """委員長が発する『○○大臣』を新 turn として追加する。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 500.0, "name": "乙田", "group": "Q党"},
    ]
    # 委員長セクション (100-500) 内で大臣をアナウンス
    cues = [
        {"start": 105.0, "end": 107.0, "text": "それでは、甲野大臣。"},
    ]
    out = inject_answerer_turns(speakers, cues)
    # 元 2 entry + 大臣 1 turn = 3 entry
    assert len(out) == 3
    # 時刻昇順
    starts = [s["start"] for s in out]
    assert starts == sorted(starts)
    # 大臣 entry
    ministers = [s for s in out if s.get("group") == "大臣"]
    assert len(ministers) == 1
    assert ministers[0]["name"] == "甲野"
    assert ministers[0].get("auto_detected") is True


def test_announcement_in_member_section_detected():
    """議員セクション内でも『○○大臣。』のように助詞無しなら検出する。

    衆議院 webtv の公式 speakers は質問者リストのみで委員長セクションが各議員
    の前に入っていないため、議員 turn 内に委員長の答弁者割り当てが埋まる。
    そこを救うのが本機能のキモなので、議員セクション内でも検出する。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 200.0, "name": "乙田", "group": "Q党"},
    ]
    # 議員 乙田 のセクション内で委員長が大臣を呼ぶ (助詞無しでアナウンス)
    cues = [
        {"start": 250.0, "end": 252.0, "text": "では、甲野大臣。"},
    ]
    out = inject_answerer_turns(speakers, cues)
    # 大臣 turn 1 件追加 → 3 entry
    auto = [s for s in out if s.get("auto_detected")]
    assert len(auto) == 1
    assert auto[0]["group"] == "大臣"


def test_duplicate_announcement_dedup():
    """同じ役職を 2 回アナウンスしても 2 個 turn が増える (時刻が違うので別 entry)。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 1000.0, "name": "乙田", "group": "Q党"},
    ]
    cues = [
        {"start": 110.0, "end": 112.0, "text": "甲野大臣。"},
        {"start": 300.0, "end": 302.0, "text": "甲野大臣。"},
    ]
    out = inject_answerer_turns(speakers, cues)
    ministers = [s for s in out if s.get("group") == "大臣"]
    # 別時刻のアナウンスは別 turn
    assert len(ministers) == 2


def test_bureaucrat_role_detected():
    """『○○局長』『○○参考人』 等の官僚系も検出される。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 600.0, "name": "乙田", "group": "Q党"},
    ]
    cues = [
        {"start": 200.0, "end": 202.0, "text": "丁原局長、答弁を求めます"},
    ]
    out = inject_answerer_turns(speakers, cues)
    bureaucrats = [s for s in out if "局長" in s.get("group", "")]
    assert len(bureaucrats) >= 1


def test_questioner_callback_not_inserted():
    """直前に大臣 turn が無い『○○くん』は新 turn として挿入しない (単独の議員指名)。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 1000.0, "name": "乙田", "group": "Q党"},
    ]
    cues = [
        {"start": 200.0, "end": 202.0, "text": "次に、乙田君"},
    ]
    out = inject_answerer_turns(speakers, cues)
    # 元 speakers のみ
    assert len(out) == 2


def test_questioner_address_in_long_cue_skipped():
    """『○○大臣、質問します』『私から○○大臣にお伺いします』のように
    長い文中で議員が大臣を呼びかけるケースは委員長アナウンスではないので除外。

    指摘: 既存実装はこれを答弁者 turn として挿入してしまい、再質問テキストが
    大臣セクションに入る誤帰属を生んでいた (回帰防止)。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 200.0, "name": "乙田", "group": "Q党"},
    ]
    # 質問者が大臣を呼びかけ + 質問本文を続ける (長い cue、call は文中)
    cues = [
        {"start": 250.0, "end": 260.0,
         "text": "まず政策の根本について、甲野大臣にお伺いしたいと思います。"
                 "今般の改正案の趣旨をご説明ください"},
    ]
    out = inject_answerer_turns(speakers, cues)
    auto = [s for s in out if s.get("auto_detected")]
    # 長文中の呼びかけは委員長アナウンスではないので新 turn を入れない
    assert len(auto) == 0


def test_return_after_minister_turn_injects_questioner_return():
    """委員長アナウンス → 大臣 turn → `_return` (○○くん) で質問者に戻る turn を挿入。

    指摘: `_return` を skip しているため、大臣 turn が次の公式 speaker まで続き
    再質問が大臣セクションに入る誤帰属が出る (回帰防止)。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 200.0, "name": "乙田太郎", "group": "Q党"},  # フルネーム
    ]
    cues = [
        # 委員長が大臣を指名 (短い cue → chair intro 判定)
        {"start": 300.0, "end": 302.0, "text": "甲野大臣。"},
        # 大臣の答弁 (本文は不要、cue は省略可)
        # 委員長が質問者に戻す (○○くん = _return)
        {"start": 500.0, "end": 502.0, "text": "乙田くん。"},
    ]
    out = inject_answerer_turns(speakers, cues)
    auto = [s for s in out if s.get("auto_detected")]
    # 大臣 1 + 質問者復帰 1 = 2 turn
    assert len(auto) == 2
    # 大臣 turn
    minister = [s for s in auto if s.get("group") == "大臣"]
    assert len(minister) == 1
    assert minister[0]["name"] == "甲野"
    # 質問者復帰 turn
    returns = [s for s in auto if s.get("return_to_questioner")]
    assert len(returns) == 1
    # フルネーム (公式 speaker の name) で戻る
    assert returns[0]["name"] == "乙田太郎"
    # 時刻順
    assert returns[0]["start"] > minister[0]["start"]


def test_return_without_prior_minister_turn_is_ignored():
    """直前に大臣 turn が無い `_return` は (単独の議員指名 / 質問順序のアナウンス
    と区別がつかないので) turn を入れない。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 1000.0, "name": "乙田太郎", "group": "Q党"},
    ]
    cues = [{"start": 200.0, "end": 202.0, "text": "乙田くん。"}]
    out = inject_answerer_turns(speakers, cues)
    auto = [s for s in out if s.get("auto_detected")]
    assert len(auto) == 0


def test_return_with_no_matching_questioner_is_ignored():
    """`_return` の姓が公式 speakers に居ないなら何もしない (誰に戻すか不明)。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長"},
        {"start": 200.0, "name": "乙田太郎", "group": "Q党"},
    ]
    cues = [
        {"start": 300.0, "end": 302.0, "text": "甲野大臣。"},
        # 公式 speakers に居ない『戊野』に戻す → 無視
        {"start": 500.0, "end": 502.0, "text": "戊野くん。"},
    ]
    out = inject_answerer_turns(speakers, cues)
    auto = [s for s in out if s.get("auto_detected")]
    # 大臣 turn のみ、質問者復帰 turn は無い
    assert len(auto) == 1
    assert auto[0].get("group") == "大臣"


def test_preserves_original_order_and_fields():
    """元 speakers の順序とフィールドは保持される。"""
    speakers = [
        {"start": 100.0, "name": "委員長甲", "group": "内閣委員長", "extra": "x"},
        {"start": 1000.0, "name": "乙田", "group": "Q党"},
    ]
    cues = [{"start": 200.0, "end": 202.0, "text": "甲野大臣"}]
    out = inject_answerer_turns(speakers, cues)
    # 元 2 entry がそのまま (フィールド保持)
    assert {"start": 100.0, "name": "委員長甲", "group": "内閣委員長", "extra": "x"} in out
    assert {"start": 1000.0, "name": "乙田", "group": "Q党"} in out


# ============================================================================
# normalize_answerer_names のテスト
# ============================================================================


from kokkai.shugiin.answerer import normalize_answerer_names


def test_normalize_unique_surname_match():
    """姓のみ検出 (1-2 字) で議員名簿に候補 1 名 → フルネームに置換。"""
    speakers = [{
        "start": 100.0, "name": "甲野", "group": "大臣", "auto_detected": True,
    }]
    members = [
        {"name": "甲野太郎"},
        {"name": "乙田次郎"},
    ]
    out, n = normalize_answerer_names(speakers, members)
    assert n == 1
    assert out[0]["name"] == "甲野太郎"
    assert out[0]["original_name"] == "甲野"


def test_normalize_multiple_candidates_skipped():
    """姓に複数候補 → 曖昧なので置換しない (false positive 回避)。"""
    speakers = [{
        "start": 100.0, "name": "甲野", "group": "大臣", "auto_detected": True,
    }]
    members = [
        {"name": "甲野太郎"},
        {"name": "甲野花子"},
    ]
    out, n = normalize_answerer_names(speakers, members)
    assert n == 0
    assert out[0]["name"] == "甲野"
    assert "original_name" not in out[0]


def test_normalize_full_name_untouched():
    """既にフルネームなら触らない。"""
    speakers = [{
        "start": 100.0, "name": "甲野太郎", "group": "大臣", "auto_detected": True,
    }]
    members = [{"name": "甲野太郎"}]
    out, n = normalize_answerer_names(speakers, members)
    assert n == 0
    assert out[0]["name"] == "甲野太郎"


def test_normalize_no_match_kept():
    """議員名簿にマッチしない姓はそのまま (政府参考人の官僚等)。"""
    speakers = [{
        "start": 100.0, "name": "丁原", "group": "局長", "auto_detected": True,
    }]
    members = [{"name": "甲野太郎"}, {"name": "乙田花子"}]
    out, n = normalize_answerer_names(speakers, members)
    assert n == 0
    assert out[0]["name"] == "丁原"


def test_normalize_skips_non_auto_detected():
    """only_auto_detected=True (既定) なら公式メタ由来の議員は触らない。"""
    speakers = [{"start": 100.0, "name": "甲野", "group": "Q党"}]
    members = [{"name": "甲野太郎"}]
    out, n = normalize_answerer_names(speakers, members)
    assert n == 0
    assert out[0]["name"] == "甲野"


def test_normalize_empty_members():
    """members が空 → 何もしない。"""
    speakers = [{
        "start": 100.0, "name": "甲野", "group": "大臣", "auto_detected": True,
    }]
    out, n = normalize_answerer_names(speakers, [])
    assert n == 0
    assert out == speakers


def test_normalize_long_partial_name_not_changed():
    """3 文字以上の不完全名 (e.g. 「甲野太」) は姓判定に乗らない。"""
    speakers = [{
        "start": 100.0, "name": "甲野太", "group": "大臣", "auto_detected": True,
    }]
    members = [{"name": "甲野太郎"}]
    out, n = normalize_answerer_names(speakers, members)
    # 3 文字は姓 (1-2 字) の範囲外なので保守的にスキップ
    assert n == 0
    assert out[0]["name"] == "甲野太"
