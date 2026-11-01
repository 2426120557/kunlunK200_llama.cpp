# Kunlun K200 非選択mixed入力分岐の除外

2026-11-01 JST。**通常token入力向けに不要なmixed分岐を構築しない方式Aを実装した。Single PDのreserve splitは3→1、実際のTG Kunlun graph実行は2→1回/step、空のCPU graph呼出しは1→0となった。数値・入力切替の検証は合格。性能はモデル・phaseで差とばらつきがあり、独立opt-inのままとする。**

## 基準・変更範囲

開始時は両repo clean。llama.cpp e39caee1824956b7d87bc969ec18770fded62dbf、library 44139408aa4045f09c2515d5c3cb0cf7badb7132。独立branch codex/k200-prune-mixed-20261101、worktree/build/resultsはprune-mixed-20261101名。libraryとdevice imageは前回の正常版を直接参照し、一切変更していない。submodule pinも同じ。

全比較はcommon Q4、Q4/F32 GET_ROWS、Activation共有、GEMV stream/N-major、GEMM Owners、SwiGLU、Attention/Flash/RoPEをON、Large PP=0、Q6 SDNN=1 / LEGACY=0。新Q6のonchip_callsとjoin_launches=0を確認した。

主な変更はsrc/llama-graph.cpp、src/llama-graph.h、examples/kunlun/prune-mixed-test.cppとCMake。GGML scheduler本体、llama-model.cpp、src/models/llama.cpp、既存kernel/API/Lease/weight ABIは変更していない。

## 原因と実装方式

Embeddingはsrc/llama-graph.cppのbuild_inp_embdで構築される。従来はtoken、外部Embedding、mixed入力をまとめて作り、ggml_build_forward_selectで必要な依存にCOMPUTEをmarkしていた。通常token入力でもmixed GET_ROWS、DUP、SET_ROWSがgraphのnode配列に残り、schedulerはCOMPUTE=0のnodeにもbackendを割り当てる。CPU backendはその演算をskipするが、CPU splitの起動・依存待ちは残っていた。

| 従来Single PD split | 内容 | 通常token時 |
|---|---|---|
| KUNLUN #0 | token Embedding GET_ROWS、mixed GET_ROWS placeholder | tokenのみactive、mixedはinactive |
| CPU #1 | inp_mixed_embdのDUP、mixed token行を挿入するSET_ROWS | 両方COMPUTE=0、実演算なし |
| KUNLUN #2 | Transformer、F32 GET_ROWS、最終FFN/RMSNorm、Q6出力 | active |

方式Bの汎用scheduler pruningは採用しなかった。allocation、backend境界、他backendへ影響する変更を避け、通常token graphから不要な分岐そのものを作らない方式Aで改善できたため。

新設定GGML_KUNLUN_PRUNE_INACTIVE_MIXEDは未設定/0/無効値で従来、明示1で対応条件内だけ有効。対象はLLAMA architecture、Kunlun常駐Q4_0 embedding、入力次元にpaddingなし、通常推論context、training/embeddings出力/pipeline parallelなし、LoRA/Hadamardなし。さらにubatch.tokenが存在し、ubatch.embd/type/embd_stateがない純粋なtoken入力に限る。

対応時はtoken ID入力と既存GET_ROWSを作り、その依存をggml_build_forward_expandで構築する。外部Embedding placeholderとmixed GET_ROWS/DUP/SET_ROWSは生成しない。通常のcompute flagをclearしたり、active依存・fence・copyを手作業で削除する処理は追加していない。

外部Embedding、mixed入力、その他の非対応条件は従来の3入力構築とforward_selectを使う。未知のモデル構成まで最適化対象に広げていない。Layer/Row Splitも現在のKunlunではpipeline parallelが無効で、代表試験では対象になった。将来pipeline parallelが有効な条件では従来経路へ戻る。

## graph再利用・入力切替・寿命

既存llm_graph_params::allow_reuseはinput形態、mixed判定、token数、sequence構成、output数等を照合する。これを維持し、llm_graph_input_embdへ構築時のKunlun policyとtoken-only状態を保存した。policy 0/1が変わったらcan_reuse=false。token-only graphにEmbedding/type/hidden-state入力が来た場合も再利用を拒否する。

再利用不可のときは既存process_ubatchのres->reset、scheduler reset、model.build_graph、sched_alloc_graphを経由して再構築・再配置する。reserveも同じ構築関数を使う。初期token-only reserveが小さくても、後からmixed入力が来たときの必要なallocationは従来の再準備経路に任せる。KVやweightの所有権はgraphの一時bufferと分離された既存管理のまま。

input token/indexの値だけが変わった場合は再利用し、set_inputsが新しい値を転送する。pointer値だけで同一内容とみなさない。実試験でtoken値変更・同じ値の再利用時はcan_reuse=1、入力形態または設定変更時は再構築を確認した。LoRA setterも既存の新map生成・sched_need_reserveで再準備され、今回adapter経路を最適化していない。

## 正確性・互換性

| 確認 | 結果 |
|---|---|
| 両モデルPP512後からgreedy8のlogits | OFF/ONでbitwise一致、NMSE/max abs/KL=0、生成token一致 |
| F16 KV保存・復元、PP→TG | 復元後next logitsがbitwise一致 |
| Tiny同一contextでtoken→外部Embedding→token→mixed→token | 全12取得位置のlogitsがOFF/ONでbitwise一致 |
| 同じtoken/index bufferの内容更新・同値再利用 | 正しい更新とgraph再利用を確認 |
| 同じcontextで設定0→1/1→0→元設定 | can_reuse=falseで再構築し、数値一致 |
| prompt7/129、ubatch64 | bitwise一致、KV復元合格 |
| PD1 / Layer / Row Split | Tiny PP128→TG4のOFF/ON logits・生成・KV復元が一致 |
| 設定未指定 | 明示0とbitwise一致、従来3split |
| CPU-only推論 | opt-in 0/1でbitwise一致、Kunlun専用構築を不適用 |

実mixed入力はpublic llama_batch_ext_add_token / add_embd / llama_processで実行した。外部Embeddingは同じGGUF Q4行からCPUで復元した値を両モードへ入力し、tokenとEmbeddingが混在する実経路を検証した。未初期化indexをわざとactive実行する試験は行っていない。大規模PPL・全モデル評価は数値一致のため不要と判断した。

## split・実行・同期・転送

以下はPP128→TG4、context1024、Single PD0の別診断。TGは実decode3回。既存audit-liteとAPI counterを再利用し、eval observerや追加の強制同期は使わない。API時間はinclusiveで、kernel・copy・waitは重なりを含むため加算しない。

| 指標 | 7B OFF→ON | Tiny OFF→ON |
|---|---:|---:|
| reserve graph node数 | 1001→998 | 691→688 |
| reserve split数 | 3→1 | 3→1 |
| graph input tensor数 | 9→6 | 9→6 |
| 実CPU graph呼出し PP / TG3回 | 1/3→0/0 | 1/3→0/0 |
| 実Kunlun graph呼出し PP / TG3回 | 2/6→1/3 | 2/6→1/3 |
| mixed由来DUP/SET_ROWSの実演算 | 0→0（inactive node自体を除去） | 同左 |
| SDK xpu_wait PP | 1672→1671 | 1257→1256 |
| SDK xpu_wait TG3回 | 5635→5632 | 4158→4155 |
| PP H2D bytes / calls | 196,896/103→196,384/102 | 197,088/169→196,576/168 |
| TG H2D bytes（3回合計）/ calls | 100,572/217→100,560/214 | 117,916/420→117,904/417 |
| PP D2H bytes / calls | 227,584/196→同じ | 196,528/136→同じ |
| TG D2H bytes（3回合計）/ calls | 390,144/588→同じ | 388,416/408→同じ |
| Single PD P2P | 0 | 0 |

非選択mixed token IDのH2DがPP128で512B、TGで4B/step減った。前回までに大きなinactive Activation copyは除去済みなので、今回はD2H容量は変わらない。必要なmetadata、既存GET_ROWSのstatus、最終logitsのCPU取得は維持する。

通常llama-benchの既存backend counterでも、Kunlun graphはPP512のwarmup込み16→8、TG128のwarmup1+128stepで258→129だった。ログ表示だけの変更ではない。Q6 dispatch・quant/SD回数、F32 GET_ROWS回数、Q4 Embedding upload数、Activation共有group数は不変。7B groupはPP506/TG8256、TinyはPP346/TG5676。

| Split代表試験（Tiny、ctx1024/PP128） | reserve OFF→ON |
|---|---:|
| Single PD1 | 3→1 |
| Layer | 4→2 |
| Row | 94→92 |

LayerのPD境界とRowの必要なP2P・同期は残した。全Splitを1graphにするために依存を削除していない。

## 通常PP512/TG128性能

同じ実行ファイル・library、Single PD0、threads6、batch/ubatch512/128、F16 KV、前述の固定設定。標準llama-bench -p512 -n128 -r1、診断・observerなし。標準benchのPP512/空KV TG128を測定。OFF,ON,ON,OFFのABBA、Tinyだけ変動が大きいためON,OFFを各1回追加。全標本を保持し、都合の悪い値を除外していない。

| モデル | phase | n OFF/ON | OFF平均±標本SD tok/s | ON平均±標本SD tok/s | 中央値 OFF / ON | 平均変化 |
|---|---|---:|---:|---:|---:|---:|
| Llama 2 7B | PP | 2/2 | 80.469 ±0.410 | 80.262 ±0.005 | 80.469 / 80.262 | -0.26% |
| Llama 2 7B | TG | 2/2 | 8.516 ±0.018 | 8.556 ±0.008 | 8.516 / 8.556 | 0.47% |
| TinyLlama | PP | 3/3 | 319.395 ±1.276 | 329.764 ±8.558 | 319.615 / 334.194 | 3.25% |
| TinyLlama | TG | 3/3 | 28.725 ±0.026 | 28.369 ±0.614 | 28.732 / 28.354 | -1.24% |

Tiny ONのPP標本は335.199/319.899/334.194、TGは27.763/28.991/28.354で変動が大きい。7B TGには小幅な上昇、Tiny PPには上昇が見られた一方、7B PPは同等域、Tiny TGの改善は確認できなかった。少数標本の差を全モデルで保証された高速化とは扱わない。

Tiny TGについて、診断では余分なQ4/Q6起動・copy・quant化の増加はなく、wait回数も減っていた。除去対象のCPU graphは空で費用が小さく、device kernel完了待ちが残るため、split半減がそのままTG倍速にはならない。単発診断ではTiny TG wait inclusive時間は89.442→84.794ms/3stepだったが、通常性能の低下原因をこれだけから断定できない。CPU空graph費用も約0.5ms/3stepに留まった。ばらつきを超えるTiny TG改善の根拠は不足しており、推測でkernelやclockを変更しない。

**構造改善とbitwise一致を確認できたのでopt-inとして採用する。Tiny TGの平均低下・変動と限定した検証範囲を踏まえ、デフォルトONへは変更しない。**

## メモリ・残存CPU・安全性

| PP128診断 | OFF | ON |
|---|---:|---:|
| 7B CPU compute buffer | 8.25 MiB | 0.25 MiB |
| Tiny CPU compute buffer | 4.25 MiB | 0.25 MiB |
| 7B GPU compute buffer | 22.38 MiB | 22.38 MiB |
| Tiny GPU compute buffer | 16.62 MiB | 16.62 MiB |
| 7B API-tracked HBM peak | 4,526,477,824 B | 4,526,478,336 B |
| Tiny API-tracked HBM peak | 780,664,064 B | 780,664,064 B |

CPU分岐用bufferは7Bで8MiB、Tinyで4MiB減った。7B HBMの差は512Bのallocation差で、weightやdevice workspaceの複製ではない。library・既存image不変を確認し、SDK/driver全内部確保を測定したとは主張しない。

通常token入力の対象構成では、Embedding→Transformer→F32選択→最終FFN/RMSNorm→Q6の途中にCPU tensor演算はない。hostによるgraph制御、KV metadata/index取得、tokenization、最終logits readbackとCPU samplingは残る。外部Embedding/mixed入力では必要なCPU分岐を残す。

ベンチ用スクリプトのkernel directoryを誤って新results側へ向けたため、初回1回だけimage読込み前のhost停止があった。固定library/image参照と存在検査へ修正し、両PD正常・errtask不変を確認、停止ラッチ・dmesg・失敗スクリプトをレビュー履歴へ保存して既知正常の小テストから再開した。device errorやresetはない。複数prompt長の保存suffixの結果照合も既存仕様へ合わせた。

最大観測60℃、80℃ thermal guard・persistent latch・他利用者確認を継承。driver/firmware/clock/OSを変更せず、正常なF32版と0.7.12を保持。モデルやSDKの複製、大量のdevice traceは行っていない。

次の候補は、Tinyの性能変動が必要な場合だけ同じ配置・処理経路で絞って追跡すること、残るkernel完了待ち/小metadata転送の費用を測ること。今回の限定pruningを汎用schedulerへ広げることや、mixed/pipeline parallelを未検証で最適化することは行わない。

## 再現・Git

libraryは44139408aa4045f09c2515d5c3cb0cf7badb7132をそのまま使用。llama.cppの機能/文書commit一覧、最終HEAD、gitlinkと固定binary情報はdelivery-summary.jsonに記録する。GitHub push、PR、main/master更新は行わない。

~~~sh
R=/home/unagidojyou/kunlun-k200
O=$R/results/prune-mixed-20261101
P=$R/worktrees/prune-mixed-20261101-llama
cmake -S "$P" -B "$R/build/prune-mixed-20261101-llama" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=ON \
  -DCMAKE_PREFIX_PATH="$R/results/getrows-f32-20261101/install" -DLLAMA_BUILD_TOOLS=ON
cmake --build "$R/build/prune-mixed-20261101-llama" -j4 \
  --target llama-bench kunlun-llm-smoke kunlun-prune-mixed-test
export GGML_KUNLUN_COMMON_Q4=1 GGML_KUNLUN_GET_ROWS=1 GGML_KUNLUN_GET_ROWS_F32=1
export GGML_KUNLUN_SWIGLU=1 KUNLUN_LARGE_PP=0 KUNLUN_Q6_K_SDNN=1 KUNLUN_Q6_K_LEGACY=0
export GGML_KUNLUN_Q4_ACT_SHARE=1 KUNLUN_Q4_MAC_DMA=1 KUNLUN_Q4_NMAJOR=1 KUNLUN_Q4_GEMM_OWNERS=1
export GGML_KUNLUN_PRUNE_INACTIVE_MIXED=1 # 未設定/0で従来graph
# library/imageは上記の保護対象installを参照。実機はthermal/latch/PD確認付きrunner経由。
python3 "$O/bench.py"       # 通常ABBA
python3 "$O/diagnostic.py"  # PP128→TG4、observer/追加同期なし
python3 "$O/regression.py"  # logits/KV/代表Split
~~~

標準bench引数は -m MODEL -p512 -n128 -r1 -b512 -ub128 -t6 -ngl99 -sm none -dev KUNLUN0 -mg0 -ctk f16 -ctv f16 -fa on -o json。正確なargvと固定環境はresults/logs/*.json、少量の集計はsummary.json。PRUNE_INACTIVE_MIXED.mdに対応条件とcache契約を記載した。
