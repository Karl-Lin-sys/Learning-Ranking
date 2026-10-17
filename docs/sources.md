# 一次資料と適用範囲

確認日: 2026-10-17（Asia/Taipei）。技術選定の3年窓は2023-10-17以降、原因分析の5年窓は2021-10-17以降。検索結果に混在したブログ、フォーラム、Wikipediaは根拠として使用していない。arXiv掲載だけを査読済みと扱わない。

## 査読済み研究

| ID | 論文・掲載先 | この成果物で支持する範囲 |
|---|---|---|
| P1 | [M3-Embedding, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.137/) | 多言語dense/sparse/multi-vector検索の候補。FindingsでありACL本会議main論文ではない。厳密にmain限定なら補助候補扱い |
| P2 | [ListT5, ACL 2024 main](https://aclanthology.org/2024.acl-long.125/) | リスト全体を比較する再ランキングの研究候補。FiDとトーナメント手順を提案。このリポジトリのlistwise lossはListT5再現ではない |
| P3 | [FIRST, EMNLP 2024 main](https://aclanthology.org/2024.emnlp-main.491/) | LLM再ランキングの推論コスト削減候補。独自データで比較するまで優劣は未決定 |
| P4 | [Unbiased Learning-to-Rank Needs Unconfounded Propensity Estimation, SIGIR 2024](https://doi.org/10.1145/3626772.3657772) / [著者の論文ページ](https://www.cse.lehigh.edu/~brian/pubs/2024/SIGIR/) | relevanceが順位とクリックの交絡要因になる。単に順位別CTRをpropensityとする危険 |
| P5 | [Fine-Grained Dynamic Framework for Bias-Variance Joint Optimization on Data Missing Not at Random, NeurIPS 2024 main](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc75254bc4b8b42f401d0ab5d6e9aa4b-Abstract-Conference.html) | 小さなpropensityで不偏推定の分散が大きくなる。clipにはバイアスと分散の交換条件がある |
| P6 | [Investigating Data Contamination in Modern Benchmarks for Large Language Models, NAACL 2024 main](https://aclanthology.org/2024.naacl-long.482/) | 事前学習と評価データの重複を監査する動機。旅行ログの時間分割手法を直接証明した論文ではない |
| P7 | [Doubly Robust Estimation for Correcting Position Bias in Click Feedback for Unbiased Learning to Rank, ACM TOIS 2023](https://doi.org/10.1145/3569453) | DRを検討する理論的背景。5年窓の原因分析に採用。3年窓の最新会議論文として数えない |

対象はテキスト検索なのでACL/EMNLP/NAACL/SIGIRを「等」に含めた。CVPR等の画像論文を無関係に追加していない。研究候補の結果を台湾旅行検索の改善率へ外挿していない。

## 公式実装とStarsのスナップショット

GitHub画面の丸め表示。2026-10-17閲覧時。全GitHubを列挙したランキングではなく、関連する公式実装の比較集合内の順序。Starsは人気の指標であり再現性、品質、因果的優位性を証明しない。

| 公式リポジトリ | 表示Stars | 判断 |
|---|---:|---|
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | 約104.8k | 実装基盤として採用 |
| [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | 約41.2k | 大規模近傍検索の候補。今回の実装はexact cosineでありFaiss未使用 |
| [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | 約12.3k | 多言語encoder候補。モデル重みは今回未取得 |
| [tensorflow/ranking](https://github.com/tensorflow/ranking) | 約2.8k | 2025-05-21 archived。新規実装の主基盤に選ばない |

[ListT5公式](https://github.com/soyoung97/ListT5)は51 Starsで、高Stars採用基準から外れるため依存に追加していない。P2の査読済み論文は研究比較に使用するが、少数Stars実装を「再現性が高く評価済み」とは呼ばない。論文と実装人気の両方を満たす候補が存在しない部分を、架空のランキングで埋めない。

## コードの公式API根拠

- [PyTorch inference_mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html): 推論のautogradを停止。eval()は別に必要。
- [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html): set_to_none=True。
- [Softplus](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softplus.html): pairwise差分に数値安定な損失。
- [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): sigmoidとBCEの安定な組み合わせ。実装ではfunctional版を使用。
- [TensorFlow Ranking overview](https://www.tensorflow.org/ranking/overview) / [BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy): 比較対象。TensorFlowを実行したという主張はしない。
- [OpenCV公式Python tutorials](https://docs.opencv.org/4.13.0/d6/d00/tutorial_py_root.html): 画像処理向け。今回のテキスト検索に不要なのでインストールしない。
- [Elastic公式RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) / [ランキング](https://www.elastic.co/docs/solutions/search/ranking): BM25とRRFの定義・検索設計。フレームワークAPI以外の検索仕様を補う公式一次資料。

閲覧時のPyTorch stableは2.14へ誘導された。実行環境は2.14.1+cpu。TensorFlow公式APIページは2.16.1と表示されたため、それを2026年の最新リリース番号と解釈していない。OpenCVは4.13.0へ誘導された。バージョンはrequirementsと実行記録で確認する。

コードは新規の教育用実装。特定の論文実装をコピーしたものでも、すべての関数が論文で効果保証されるものでもない。Big-Oはここにあるループ・配列の構造から導出した。未確認の「解決済みIssue」を根拠にした修正や、メモリリーク皆無という主張はない。

## 台湾法の一次資料（ML論文とは別の要請）

- [個人資料保護法・全國法規資料庫](https://law.moj.gov.tw/LawClass/LawAll.aspx?PCode=I0050021): 目的、収集・処理・利用の根拠、告知、安全管理等を実データ導入前に確認する。19・20条を参照。改正条文には施行日が別途指定されるものがあり、改正日だけで施行済みと断定しない。
- [著作權法・全國法規資料庫](https://law.moj.gov.tw/LawClass/LawAll.aspx?PCode=J0070017) / [智慧財產局10條之1](https://www.tipo.gov.tw/tw/copyright/694-17731.html): 思想・原理と具体的表現を区別する。論文の図表、全文、他者コードの無条件転載を正当化する条文ではない。

この公開物は合成データ、新規コード、短い独自要約と出典リンクだけを収録する。証明書、身分証番号、学籍番号、第三者の氏名、顧客ログ、求人全文は収録しない。ライセンス・個人情報の確認は具体的な利用条件次第であり、この資料を包括的な台湾法適合の鑑定書としない。
