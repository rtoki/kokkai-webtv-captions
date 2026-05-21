"""README 用のスクリーンショット用 sample HTML を生成する。

実議員名を含まない架空データで sangiin.render.render_html を呼び、
docs/sample.html を出力する。Chrome headless でスクリーンショットを
取るのは別途シェルから。
"""

from __future__ import annotations

from pathlib import Path

from kokkai.sangiin.render import render_html


SAMPLE = {
    "date": "2099-01-01",
    "title": "サンプル委員会",
    "sections": [
        {
            "start": 0.0,
            "name": "甲山一郎",
            "group": "委員長",
            "text": "",
            "turns": [
                {
                    "klass": "chair",
                    "role": "委員長",
                    "speaker": "甲山一郎",
                    "text": "これより会議を開きます。本日の議題に入ります。"
                    "質疑者の発言を許します。",
                },
            ],
        },
        {
            "start": 60.0,
            "name": "乙田二郎",
            "group": "架空党 A",
            "text": "",
            "turns": [
                {
                    "klass": "questioner",
                    "role": "議員",
                    "speaker": "乙田二郎",
                    "text": "それでは、本案の趣旨について大臣にお伺いします。"
                    "今回の改正案では、対象範囲が拡大されると伺っていますが、"
                    "その背景にある問題意識を改めてご説明いただけますでしょうか。",
                },
            ],
        },
        {
            "start": 240.0,
            "name": "丙山三郎",
            "group": "大臣",
            "text": "",
            "turns": [
                {
                    "klass": "minister",
                    "role": "大臣",
                    "speaker": "丙山三郎",
                    "text": "お答えいたします。ご指摘のとおり、近年、関連する事案"
                    "が増加していることを踏まえまして、対象範囲の見直しを行った"
                    "ものでございます。詳細については政府参考人からも補足させて"
                    "いただきます。",
                },
            ],
        },
        {
            "start": 480.0,
            "name": "丁川四郎",
            "group": "政府参考人",
            "text": "",
            "turns": [
                {
                    "klass": "bureaucrat",
                    "role": "政府参考人",
                    "speaker": "丁川四郎",
                    "text": "技術的な観点から補足いたします。"
                    "改正後の運用については、関係機関と連携しながら段階的に"
                    "導入してまいる予定でございます。",
                },
            ],
        },
        {
            "start": 720.0,
            "name": "戊田五郎",
            "group": "架空党 B",
            "text": "",
            "turns": [
                {
                    "klass": "questioner",
                    "role": "議員",
                    "speaker": "戊田五郎",
                    "text": "ありがとうございます。"
                    "続いて、関連する別の論点についてお尋ねします。",
                },
            ],
        },
    ],
}


def main() -> None:
    html = render_html(SAMPLE)
    out = Path("docs/sample.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
