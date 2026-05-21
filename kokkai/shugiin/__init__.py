"""
衆議院インターネット審議中継 (shugiintv.go.jp) から発言者別 HTML を生成するモジュール。

Phase 2 (現状): 発言者リスト + HLS URL を取得し、字幕無しの「発言者タイムライン HTML」を出力。
Phase 3 (予定): mlx-qwen3-asr で音声を文字起こしして、発言テキスト付き HTML に格上げ。
"""
