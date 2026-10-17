# 技術選定と検証設計

## 結論

最初の比較対象は「販売可能条件のフィルタ＋BM25」。次に独立したdense召回を加えてRRFで統合し、小さなPyTorch LTRへ進む。LLM再ランキングとRAGは、検索品質とコストの比較が済んでから別の実験として評価する。これは求人要件から立てた設計判断であり、実測した会社システムの診断ではない。

実装は教育用に小さくした。各route上位20件、MLP候補最大20件、API上位最大20件、文字列最大256文字、request最大4096 bytes。大規模本番性能を証明する数値ではない。

## 技術比較

| 技術 | 長所・採用理由 | 構造的な限界・検証 |
|---|---|---|
| BM25 / 倒排索引 | 語彙一致の説明が容易。安い基準 | 同義語・自然文の意図を取り逃す。今回CJK文字単位であり本格分詞は未実装 |
| BGE-M3候補 | 多言語・複数retrieval形式を扱える[P1] | 重み、ライセンス、繁體中文と地名、学習データ重複を別途確認。今回未ダウンロード |
| Exact cosine | ANNの正解集合を確認できる | O(ND)で全商品走査。今回手作業ベクトルで接続のみ検証 |
| Faiss / ANN候補 | 高Stars公式実装。候補検索をスケールさせる候補 | ANN Recallと遅延の交換条件、削除更新、属性フィルタを測定。ANNを無条件O(log N)と扱わない |
| RRF | 異なるスコア尺度を順位で統合 | 同じ結果を重複加点しない。切り捨て後の未取得商品は復元できない。constant=60は固定基準、最適値ではない |
| Pointwise | 独立の二値BCEで小さい基準 | 0/1化で段階評価情報を失う。負例多数ならaccuracyだけが高くなる |
| Pairwise | 同じqueryの良い商品が悪い商品を上回るように学習 | 全pairはO(M²)。クリック正例対未クリックを無条件に真の選好としない |
| Listwise | query内のgain分布をsoftmaxで学習 | listの構成に依存。今回の損失はNDCGそのものでもListT5でもない |
| ListT5 / FIRST研究候補 | リスト比較・推論効率改善を研究[P2,P3] | 高Stars条件を満たす実装と区別。GPUコスト・繁體中文・順位頑健性を検証するまで採用保留 |
| PyTorch | 今回の学習・推論の採用基盤 | バッチ上限、eval、inference_mode、依存固定を実装。速度・メモリは環境依存 |
| TensorFlow Ranking | 既存TF資産がある場合の比較対象 | 公式repoはarchived。TF本体の廃止という意味ではない |
| OpenCV | 写真検索を追加する場合の画像処理候補 | 今回はテキスト中心なので依存不要。求人のツール欄を全部入れることを目的にしない |

Elasticsearch/OpenSearch/Solrの実システムは提供されていないため、エンジン間の性能順位を捏造しない。既存運用、分析器、filter付きANN、索引更新、ライセンス、可観測性、同一負荷のp95/p99を基準に選ぶ。本作品では公式Elastic仕様を設計の参照にし、独立した小型基準を実装した。[出典一覧](sources.md)

## 原因仮説1：順位を「人気」と誤認する選択・表示バイアス

**仮説：** 上に出た商品だけが見られ、クリックされる。過去順位に依存する特徴がさらに上位商品を優遇する。ログだけでは不人気なのか未観測なのかを区別できない。[P4]

**確認する記録：** impression全候補、表示位置、実際のexposure、logging policy version、同じ意図内のpositionと関連度、device、時間。クリックだけを抽出したdatasetでは検証しない。適法な範囲で少量のランダム化表示を行い、割当確率を記録する。

**解決策1：** human-judged relevanceによる基準を作り、既知または識別可能な観測確率で補正する。単純なposition-based modelでC=R×E、Eの確率pが適切に分かるならE[C/p]=E[R]となる。ただし交絡、trust bias、クリック誤認があるとこの等式は使えない。rank別CTRをpと決めつけない。[P4,P7]

`ips_mean`はこの限定された仮定を合成実験で確認する機能。clip(1/p)で分散を抑えるが不偏性を失う。ESSとclip率を出し、clip=5/10/20等の感度をtrain/validationで確認する。p=0を拒否するが、それで未露出商品のsupportが作られるわけではない。DRも前提を満たすpropensityまたはoutcome model、適切なcross-fitting等が必要で、今回は実装していない。[P5,P7]

**反証・採用条件：** 補正後の結果がランダム化／独立評価と整合するか、改善の信頼区間、重みの分散を調べる。合成実験で成立しても本番補正成功とは言わない。

## 原因仮説2：未来情報とモデル選択で離線成績が高く見える

**仮説：** query行を無作為分割して同一sessionが混じる、テスト期間の予約数が特徴に入る、予約成立前に負例確定する、encoder事前学習と評価が重なる。モデルをtestで繰り返し選ぶとtestにも過学習する。事前学習汚染の問題は[P6]が動機であり、以下の時間設計は本実装の監査規則。

**解決策2：** query impressionを丸ごと時間分割し、区切りを跨ぐsessionを全partitionから除く。feature_time<=impression_timeを必須にし、label_time<各partitionの締切まで待つ。前処理の平均・標準偏差はtrainだけでfit。early stoppingはvalidation loss、方式選択はvalidation NDCG、testは選択後1回。モデルのseedとデータ生成seedを固定する。

実装は時刻を信用できるmetadataとして検査する。間違ったfeature_timeが付いた未来集計を自動検出するわけではない。本番ではpoint-in-time joinのSQL、集計元のevent_time、重複IDを監査する。商品ID共有は既存商品への将来検索で必ずしも漏洩ではない。未知user・未知商品・新しい季節を測る独立holdoutを追加する。encoderの訓練集合が不明なら「汚染排除済み」と言わない。

**反証・採用条件：** 時間分割・entity holdout・重複除去前後で差を比較。testを見て変えた場合は新しい未使用期間を確保。rolling-window実験と複数seedを追加し、スコアが安定するまで業務改善を主張しない。

## 原因仮説3：召回漏れと評価母集団が、精排の改善に隠れる

**仮説：** 目的地や日付が合う商品が候補に入らない。候補だけでIDCGを作ると漏れを罰せず、NDCGが良く見える。人気queryが集計を支配し、MRRやCTRが飽和して品質差を隠す。[P1,P2]はretrieval/reranking改善候補を示すが、この会社で原因が発生した証拠ではない。

**解決策3：** 日付、出発地、人数、販売可否、通貨・価格単位などの厳密条件を検索の各routeへ適用し、lexicalとdenseの和集合を融合する。まずeligible corpus全体のjudgmentでRecall@Kを測り、その後同じpoolのIDCGでNDCG@K、MRR@Kを測る。候補を落とした結果を「非関連」に書き換えない。本実装はunknown judgmentを例外にし、全0 relevance queryは0点として母数と件数を明示する。

**反証・採用条件：** 同一query集合でBM25のみ→denseのみ→RRF→同じ候補へのrerankをablation。routeごとの候補Recall、zero-result率、head/tail、言語、商品種別、device、新規商品を別表示する。合成head/tailは`qid % 4`のラベルであり、実際の頻度や長尾性を再現していない。少数sliceは区間が広いことも報告する。

class imbalanceにはaccuracy単独を使わず、必要なら元の本番prevalenceでPR曲線・calibrationを評価する。訓練negative downsamplingを行うなら選択確率を保存し、評価集合をdownsampleしない。未クリック＝負例と固定しない。最適な関連度順位と予約確率順位は同じとは限らない。

## オンライン評価と運用への接続

1. ユーザー等の安定した単位でA/Bを割当。`experiment_arm`は合成IDの固定割当例。hash化は匿名化保証ではない。
2. 主指標を「割当利用者当たり、所定期間内に予約した割合」と定義。ITT母集団を保ち、未クリック利用者を除外しない。キャンセルを含むnet conversion、価格、露出量も確認。
3. 効果最小幅MDE、必要数、期間、α、power、主指標、guardrail、終了条件を事前登録する。固定期間中の繰り返し覗き見を通常のp値で判定しない。
4. `analyze_ab`は成熟したunit単位の二値outcomeを集計し、bootstrap区間と50:50 SRM近似を計算。多重比較、疎な全0データ、ネットワーク効果、干渉、季節性を解決する機能ではない。CIだけで自動採用しない。
5. p95/p99、エラー率、zero-result率、キャンセル率、販売不可商品率、feature欠損、分布変化、index freshnessを監視。`log_summary`は固定windowの一部のみ。閾値は既存SLOと実測から決める。

これらは統計・運用上の明示的な設計であり、特定の論文を再現したオンライン実験ではない。実顧客がいないためA/B結果を捏造しない。

本番拡張では、商品eventにversion/event_timeを持たせ冪等upsert→新indexにbuild→品質確認→alias切替とrollbackを行う。query cacheはnormalized query・filter・locale・inventory/model/index versionをkeyにし、容量上限とTTLを設ける。価格/在庫は回答前にsource of truthへ再照会する。これらの分散機構はこのrepoでは設計までであり実装済みとしない。

API demoのHTTPServerは単一処理で、認証・TLS・水平スケール・gateway・rate limitを備えない。本番ではmodelをworker起動時に一度ロードし、queueとdeadlineを制限、reranker timeout時に既存RRFへfallbackする。候補・token・batchを制限し、全pair M²を無制限に作らない。RAGは取得商品IDと出典にgroundし、価格を生成せず現在の取得値を使い、資料中の命令を実行しない。

## メモリと性能

学習履歴にはPython floatのみを保存し、計算graphを保持しない。inference_modeとevalを併用し、zero_grad(set_to_none=True)とbounded batchを使用。best checkpointは一つだけ保持。デモの小型indexは各検索で再構築するため、その費用もlocal latencyへ含めている。本番では更新versionごとのimmutable indexを再利用する。

「メモリリークを排除済み」という証明は行っていない。長時間負荷試験、RSS/VRAMのwarmup後の傾向、concurrency、OOM回復は未検証。空きcacheと到達不能memoryの漏洩を同一視しない。cacheを毎request空にして高速化したという主張もない。

参照IDは[sources.md](sources.md)。Big-Oは[complexity.md](complexity.md)。
