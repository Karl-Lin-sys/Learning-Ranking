# 各関数のBig-O

実装のループ・保存配列からの導出。論文の性能保証ではない。Pythonのhash table操作は平均O(1)と仮定。sortはO(n log n)。入力・出力サイズが0/1のときlog項は定数と解釈する。

記号：L=文字長、T=全token数、N=商品数、D=vector次元、P=query語のposting総数、M=1 queryの候補数、U=route和集合のunique数、R=route結果総数、J=judged pool数、Q=query数、F=特徴数、H=隠れ層幅、B=bootstrap反復数、E=epoch数、b=training batch、S=JSON bytes。空間は特記しない限り追加入力外の確保を含む。

| ファイル・関数 | 時間 | 空間 | 注意 |
|---|---|---|---|
| core.tokens | O(L) | O(L) | lower-case＋正規表現。CJKは1文字単位 |
| Product生成 | O(1) | O(1) | 既存文字列・tupleへの参照。入力を含めればO(L+D) |
| eligible | O(N) | O(N) | 条件検査と参照リスト |
| BM25.__init__ | O(T+N) | O(T+N) | title token化、posting、長さ |
| BM25.search | O(L+P+M log M) | O(L+M) | top-k heapではなく候補全sort |
| dense_search | O(ND+N log N) | O(N) | query/product normも毎回計算。ANNではない |
| rrf | O(R+U log U) | O(U) | routeごとdedupe、全unique結果sort |
| metrics | O(M+J log J) | O(M+J) | ranking検査とIDCG sort。上位kのみの分母ではない |
| temporal_split | O(Q) | O(Q) | rows自体はcopyせず参照、session跨ぎ検査 |
| ips_mean | O(N) | O(N) | weights保持。重み合計からESS算出 |
| paired_bootstrap | O(BQ+B log B) | O(B+Q) | Q個の独立delta。反復ごとの選択は逐次 |
| experiment_arm | O(L) | O(L) | experimentとunitの連結hash |
| model.make_model | O(FH) | O(FH) | 1 hidden MLP。H固定ではFに線形 |
| ranking_loss(pointwise) | O(bM) | O(bM) | logits BCE。forwardモデル費用は別 |
| ranking_loss(pairwise) | O(bM²) | O(bM²) | 差分・mask・損失を全pair生成。b/Mの上限が必要 |
| ranking_loss(listwise) | O(bM) | O(bM) | gain正規化、log_softmax |
| tensors | O(QMF) | O(QMF) | 小規模全datasetをCPU上にmaterialize |
| fit | O(E[(Qt+Qv)MFH+(Qt+Qv)C]) | O((Qt+Qv)MF+FH+bMH+QvMH+max(b,Qv)C) | C=M²(pairwise)、その他M。validationは全件一括。best state・Adam stateはO(FH) |
| predict | O(QMFH) | O(QMF+QMH) | 出力のPython list O(QM)も含む |
| demo.synthetic_queries | O(QM) | O(QM) | F=6固定。一般化すればO(QMF) |
| catalog | O(1) | O(1) | 6件固定。一般化はO(N(L+D)) |
| search | O(T+ND+N log N+R+U log U+UFH) | O(T+ND+U(F+H)) | modelあり時。F,H,D固定、index再構築も含む。実際のcatalogは6件 |
| evaluate | O(QM log M) | O(QM) | qrels全M、macro集計 |
| run | 3方式fit費用＋test O(QM(FH+log M))＋bootstrap O(BQ+B log B)＋A/B O(BNa+B log B)＋反復search | dataset・model・集計の最大保持分 | bias生成10,000、latency反復100、A/B unit1,000は固定 |
| main | run費用＋O(S) | run空間＋O(S) | JSON文字列とファイル出力 |
| Handler.do_GET | O(1) | O(1) | 固定health response |
| Handler.do_POST | O(S+search) | O(S+search) | S<=4096、query<=256、vector次元3 |
| Handler.respond | O(S) | O(S) | JSON encode |
| Handler.log_message | O(1) | O(1) | データを保存しない |
| analyze_ab | O(BN+B log B) | O(N+B) | 独立arm bootstrap。Nはassigned unit数 |
| log_summary | O(N log N) | O(N) | 固定windowのlatency sort |

validation全件一括とdataset全materializeは小さい実験向け。本番訓練へ拡大する前にDataLoader/streamingとvalidation batch化が必要。`inference_mode`はgraphを保存しないが、入力や出力配列が無料になるわけではない。計算量から実機のms、ピークRAM、GPU throughputは推定せず測定する。

テスト関数は入力サイズを固定したfixtureで、各被テスト関数の上記費用を持つ。HTTP待ち・thread shutdown・I/O時間はCPU計算量の外にある。`__init__.py`は宣言のみで追加の業務関数なし。
