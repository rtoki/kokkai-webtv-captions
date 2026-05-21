"""kokkai.list — 両院の中継一覧を取得して、取込済 / 未取込を判定して表示。

ユーザーが sid / deli_id を覚えていなくても、CLI から「最近どんな会議が
あったか」を見つけて取り込めるようにするためのパッケージ。
"""

__all__ = ["sangiin_list", "shugiin_list", "status", "render"]
