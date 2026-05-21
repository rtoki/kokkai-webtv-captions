"""extract.py の純関数 (ネットなし) のテスト."""

from __future__ import annotations

import json
import textwrap

import pytest

from kokkai.sangiin.extract import (
    _extract_m3u8_url,
    _ts_to_sec,
    find_subtitle_playlist,
    parse_vtt_cues,
    vtt_to_text,
)


# ---------- _extract_m3u8_url (VOD / LIVE 両対応) ----------

def test_extract_m3u8_url_vod():
    """VOD: player JS の ``url:"...m3u8"`` を直接利用、is_live=False。"""
    js = (
        'var s={video_info:[{url:"https://example.com/path/index.m3u8",'
        'isLive:!1}],channel_info:{url:""}};'
    )
    m3u8, is_live = _extract_m3u8_url(js)
    assert m3u8 == "https://example.com/path/index.m3u8"
    assert is_live is False


def test_extract_m3u8_url_live(monkeypatch):
    """LIVE: ``url:""`` で、``channel_info.url`` の JSON から HLS を組み立てる。"""
    js = (
        'var s={video_info:[{url:"",isLive:!0,captionUrl:""}],'
        'channel_info:{url:"https://live.example.com/live/AAA/BBB/channel-info.json"}};'
    )

    def fake_http_get(url):
        assert url == "https://live.example.com/live/AAA/BBB/channel-info.json"
        return json.dumps({
            "manifest": "index.m3u8",
            "status": {"output": "active", "subtitle": "active"},
        })

    monkeypatch.setattr("kokkai.sangiin.extract.http_get", fake_http_get)
    m3u8, is_live = _extract_m3u8_url(js)
    assert m3u8 == "https://live.example.com/live/AAA/BBB/index.m3u8"
    assert is_live is True


def test_extract_m3u8_url_no_match():
    """player JS に url も channel_info も無い → SystemExit。"""
    js = 'var s={video_info:[{poster:"x"}]};'
    with pytest.raises(SystemExit):
        _extract_m3u8_url(js)


def test_ts_to_sec():
    assert _ts_to_sec("00:00:00.000") == 0.0
    assert _ts_to_sec("00:00:30.500") == 30.5
    assert _ts_to_sec("01:02:03.000") == 3723.0
    assert _ts_to_sec("00:01:00.000") == 60.0


def test_find_subtitle_playlist_present():
    master = textwrap.dedent("""
        #EXTM3U
        #EXT-X-VERSION:10
        #EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="Japanese",DEFAULT=NO,AUTOSELECT=YES,FORCED=NO,LANGUAGE="ja",URI="index-9.webvtt.m3u8"
        #EXT-X-STREAM-INF:BANDWIDTH=1160000,SUBTITLES="subs"
        index-0.m3u8
    """).strip()
    url = find_subtitle_playlist(
        "https://example.com/path/index.m3u8", master
    )
    assert url == "https://example.com/path/index-9.webvtt.m3u8"


def test_find_subtitle_playlist_absent():
    master = textwrap.dedent("""
        #EXTM3U
        #EXT-X-STREAM-INF:BANDWIDTH=1213932
        720p.m3u8
    """).strip()
    assert find_subtitle_playlist("https://x.test/a.m3u8", master) is None


def test_parse_vtt_cues_basic():
    vtt = textwrap.dedent("""
        WEBVTT

        00:00:01.000 --> 00:00:03.000
        こんにちは

        00:00:03.500 --> 00:00:05.000
        本日はよろしくお願いいたします
    """).strip()
    cues = parse_vtt_cues(vtt)
    assert len(cues) == 2
    assert cues[0]["start"] == 1.0
    assert cues[0]["text"] == "こんにちは"
    assert cues[1]["text"] == "本日はよろしくお願いいたします"


def test_parse_vtt_cues_dedup_rolling():
    """HLS のローリングバッファ字幕の重複が除去される。"""
    vtt = textwrap.dedent("""
        WEBVTT

        00:00:01.000 --> 00:00:03.000
        こんにちは本日はよろしくお願いします

        00:00:02.000 --> 00:00:04.000
        本日はよろしくお願いします次の議題に入ります

        00:00:03.000 --> 00:00:05.000
        次の議題に入ります
    """).strip()
    cues = parse_vtt_cues(vtt)
    joined = " ".join(c["text"] for c in cues)
    # 同一フレーズが3回連続出現しないことを確認
    assert joined.count("本日はよろしくお願いします") == 1
    assert joined.count("次の議題に入ります") == 1


def test_vtt_to_text():
    vtt = textwrap.dedent("""
        WEBVTT

        00:00:01.000 --> 00:00:03.000
        一行目

        00:00:03.500 --> 00:00:05.000
        二行目
    """).strip()
    text = vtt_to_text(vtt)
    assert "一行目" in text
    assert "二行目" in text
    assert "-->" not in text
