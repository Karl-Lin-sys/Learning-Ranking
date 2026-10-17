# Learning-Ranking：旅行搜尋與排序實驗室

旅行検索を「探す → 候補を混ぜる → 並べる → 正しく測る」に分けた、面接で説明できるPythonプロトタイプです。会社の本番システム、採用保証、実利用者の改善実績ではありません。

**合成データのみ。** 実顧客ログ、資格証明書、個人識別情報、求人全文、他者のモデル重みは含みません。結果が良くても旅行検索で同じ効果が出るとは限りません。

## すぐ動かす

Python 3.12とCPUを使用します。仮想環境を有効化して実行してください。

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m unittest discover -s tests -v
python -m travel_rank.demo
python -m travel_rank.api
```

`reports/demo.json`に学習・評価の結果が出ます。APIは`127.0.0.1:8000`にだけbindします。別のターミナルで：

```python
import json
from urllib.request import Request, urlopen

request = Request(
    'http://127.0.0.1:8000/search',
    data=json.dumps({'query': '親子 飯店', 'vector': [0.9, 0.1, 0.4],
                     'destination': '台北', 'budget': 4000, 'k': 3}).encode(),
    headers={'Content-Type': 'application/json'},
)
with urlopen(request, timeout=5) as response:
    print(json.load(response))
```

この3次元vectorは手作業の例で、自然言語Embeddingではありません。APIはBM25＋exact cosine＋RRFを返し、デモ実験では追加で学習済みMLPを接続します。学習用の人工特徴と検索用特徴の分布は一致していないため、接続結果を品質評価に使いません。

## 実装と読む順序

| 内容 | ファイル |
|---|---|
| 日本語の技術比較、原因仮説3つ、評価・運用設計 | [設計資料](docs/design_ja.md) |
| 幼児向けの説明、日本語・繁體中文の面接練習 | [面接ガイド](docs/interview.md) |
| 査読確認、公式API、GitHub Stars、台湾法出典 | [出典](docs/sources.md) |
| 全業務関数の時間・空間計算量 | [Big-O](docs/complexity.md) |
| フィルタ、BM25、cosine、RRF、NDCG、MRR、Recall、時間分割、IPS | [core.py](travel_rank/core.py) |
| Pointwise / Pairwise / Listwise、train-only正規化、early stopping | [model.py](travel_rank/model.py) |
| 合成実験、候補検索、順位付け、bootstrap、ローカル速度計測 | [demo.py](travel_rank/demo.py) |
| bounded request、health、検索API | [api.py](travel_rank/api.py) |
| A/B割当後の集計、SRM、CVR、検索ログ監視 | [operations.py](travel_rank/operations.py) |
| 実行済みの条件と制限 | [検証記録](reports/validation.md) |

## 証明できること・できないこと

合成データ上での手順の動作、漏洩検出、メトリクス境界条件、API契約をテストしています。学習は220 query、validationは70、testは70です。モデル選択はvalidationのみ。testでモデルを選び直しません。

実トラフィックのA/B、Elasticクラスタ、事前学習encoder、GPU負荷試験、分散ストリーム、認証・TLS・本番デプロイは未実施です。CVR改善、メモリリーク不在、全バイアス排除、3年以上の実務経験はこの作品から証明できません。MRRが飽和することも含め、合成テストの限界を説明してください。

BM25やRRFは新しい発明と主張しません。研究候補は2023-10-17以降の査読済み論文、APIは公式資料で確認しました。著作権・個人情報については[出典と公開範囲](docs/sources.md)を参照してください。
