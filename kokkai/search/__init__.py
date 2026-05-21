"""
kokkai.search — 文字起こし済みの会議に対する BM25 全文検索。

このパッケージは「DB を使わない、毎クエリで out/ ディレクトリをスキャンする」
設計で、SQLite 等の永続インデックスは作成しない。

検索対象:
  out/*.meta.json + 同名 .vtt (sangiin) または同名 _transcript.json (shugiin)

国会会議録 (kokkai.ndl.go.jp) への確定収録は本会議で 1-2 ヶ月、委員会で 3-5 週間
かかるため、本ツールの主戦場は「直近 60 日程度の未収録期間」になる。それ以上
古い会議は公式の国会会議録検索に出てくるので、結果に併記する。
"""

__all__ = ["index", "query", "tokenize", "render"]
