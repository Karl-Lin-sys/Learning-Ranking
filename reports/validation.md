# 実行検証

実施日：2026-10-17（Asia/Taipei）。Windows、Python 3.12.14、PyTorch 2.14.1+cpu、NumPy 2.5.4。デモはCPU 1 thread、モデルseed=7、合成データseed=23。GPU・他OSの実行は未確認。

## 実行したコマンド

```sh
python -m unittest discover -s tests -v
python -m travel_rank.demo
```

**18 tests passed**。最終テスト実行のunittest計測は4.045秒。APIのlocalhost HTTP送受信、異常body、長すぎるbody、未来特徴、跨期session、遅延label、重複query、未判定候補、duplicate結果、zero vector、IPS support、lossの順位方向、推論、A/B集計・未成熟outcome、監視集計を含む。

依存追加前にNumPy不足の警告が出たためNumPyを追加し、上記の最終テストを再実行した。これは一次Issueでメモリリーク修正を証明したものではない。依存は`constraints-tested.txt`に記録。

## 合成実験結果

train/validation/test = **220 / 70 / 70 queries**、各12候補。以下は全て作り物のデータ。完全な記録は[demo.json](demo.json)。

| validationの方式 | NDCG@5 | epoch数 |
|---|---:|---:|
| pointwise | 0.963080 | 19 |
| pairwise | 0.963025 | 35 |
| listwise | 0.964792 | 30 |

validationだけでlistwiseを選択。選択したモデルのtest NDCG@5は**0.955208**、MRR@5は1.0、Recall@5は0.705082。第0特徴だけの単純baseline NDCG@5は0.774024。queryごとのpaired bootstrap差分95%区間は[0.141446, 0.220559]。これは固定した合成モデル・query標本についての区間で、実業務効果やモデル学習のseed変動を含まない。

MRR=1はこの課題が容易であることを示す。召回システム間の比較でも、本番のsemantic model benchmarkでもない。trainingのfeature分布と検索デモのadapterは異なり、モデルを接続できたことと、その検索が良くなったことを区別する。

examinationが独立という合成バイアス実験：真の関連率0.3、raw CTR 0.1717、IPS 0.2961。実ログへこの数値や仮定をそのまま使わない。

同じCVRを与えた合成A/B null実験の区間は0を跨いだ。実利用者実験は一切実施していない。ローカルlatencyは6商品・逐次100回・warmup5回だけ。サーバHTTP latency、並列処理、大規模index、ネットワーク、embedding生成は含まない。JSON内に環境依存の実測値を残している。

## 公開前チェックと残る範囲

証明書ファイル、識別番号、検証コード、顧客データ、求人全文はrepoへ追加していない。コードのcompile、Git差分の空白検査、公開ファイル一覧の確認を実施。CI設定は用意したが、この記録の18件はローカル結果。GitHub Actions成功と同一視しない。

未実施：実トラフィックA/B、事前学習モデル評価、ANN、Elastic/OpenSearch/Solr接続、本番deploy、負荷・長時間メモリ試験、依存脆弱性監査、法的鑑定。これらを完了済みと記載しない。
