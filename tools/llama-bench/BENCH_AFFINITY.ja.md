# Benchmarkのmain-thread affinity

CPU affinityは明示opt-inで、通常推論では変更しない。既存の`--cpu-mask`はworker用で、本機能と役割が異なる。

## 明示指定

```bash
LLAMA_BENCH_MAIN_CPU=2 GGML_KUNLUN_Q4_GATE_UP_PAIR=0 \
  ./llama-bench -m <GGUF> -p 512 -n 128 -r 3 \
  -t 6 -b 512 -ub 128 -ctk f16 -ctv f16 -fa on \
  -ngl 99 -sm none -dev KUNLUN0 -mg 0 -o json
```

次の独立processではPairだけを1にする。今回のi7-8700ではCPU2を基準とするが、実装にはCPU2をハードコードしていない。`LLAMA_BENCH_MAIN_CPU`は未設定/`off`/`none`で無固定。数値`0`はCPU0の指定でありOFFではない。LinuxでCPU_SETSIZE未満の許可CPUを扱い、無効な明示指定では計時前に失敗する。

workerの作成と標準warmup（同一context対比較では両modeの全workload warmupも）が済んでから、呼出しmain threadのみを固定する。IRQ affinity、governor、priority、OS全体の設定は変えない。計測後に元のmaskを復元する。復元はmaskの復元であり、直前と同じ実行CPUへ戻すことの保証ではない。既存workerは変更しないが、一般に固定後に新しく作られたthreadはOSのaffinity継承規則に従う。

## topology/IRQ確認とauto

```bash
python3 tools/llama-bench/bench_affinity.py --main-cpu auto --inspect \
  --json-output <新しいtopology.json>
python3 tools/llama-bench/bench_affinity.py --main-cpu 2 -- \
  <llama-bench> <同じ引数>
```

Python helperはlauncher自体を固定しない。許可mask、online CPU、package/core、SMT siblingとXPU IRQのeffective affinityを調べ、選択CPUをbenchmarkへ渡す。autoではXPU IRQの物理core全体を除外し、許可された各physical coreの代表から最小CPU番号を選ぶ。IRQ CPUが許可mask外でもそのsiblingは除外する。IRQ/topologyが不明、または安全な候補がない場合は、理由を出して無固定へ戻る。明示指定なら不完全な調査結果やIRQ coreとの共有を警告し、許可mask内の指定を尊重する。

`/proc/interrupts`の`klN_devN` / Kunlun / XPUを識別対象とする。IRQのrequested maskをeffective maskと同一視しない。識別できないdriver名はauto無固定となるため、記録を確認して明示指定する。再起動後もIRQ番号や配置が同じとは仮定しない。

`BENCH_AFFINITY_*`は計時外で、設定前後・復元後のmask/実行CPU、topology、IRQ配置を記録する。`bench_affinity.prepare(policy, environment)`は既存runnerへ組み込める。今回のguard付きrunnerでは`LLAMA_BENCH_MAIN_CPU_POLICY=2|auto|off`を受け取り、各processの`*-affinity.json`へ保存する。OFF/ONには同じpolicyを渡す。

通常性能ではPairの詳細stats、SDK timer、allocation traceを無効にする。CPU/PD/thermal guardは外さない。同一processの反復を独立標本として数えない。

## Benchmark限定の診断

- `K200_BENCH_SWITCH=GGML_KUNLUN_Q4_GATE_UP_PAIR`、`K200_BENCH_SEQUENCE=0110`（または1001）と`-r 4`は、両modeをprewarmした同一contextの対比較。専用per-trial記録を使用し、mode混在の通常JSON平均を比較値にしない。
- `-p 0 -n 0 -pg 512,128`は実PP後にKVを保持してTGを継続する。独立`-p 512 -n 128`とは別条件。
- `LLAMA_BENCH_SPLIT_TIMING=1`はPP/TG境界のHost clockを追加して連続実行を分割記録する。通常の独立測定ではOFFにする。totalとその内訳を重複加算しない。

## 再利用・install

`bench-main-affinity.h`はSDK非依存のbenchmark用RAII helper。standalone GEMV/GEMM benchmarkでも、worker/prewarm後に`bench_main_affinity scope; scope.apply_from_env();`、計時後に`scope.restore();`を呼ぶ。戻り値を確認する。推論backendへは組み込まない。

LLAMA_TOOLS_INSTALLが有効なinstallでは、`llama-bench-affinity`をbin、headerを`include/llama-bench`へ配置する。Pythonはauto/discovery launcherにだけ必要で、直接の数値指定には不要。通常のprivate buildではsourceのhelperも利用できる。実験用installで確認し、既存installを上書きしない。

### Guard付きrunnerの再利用

`kunlun_guarded_runner.configure(workspace, results, source_roots, check_users=...)`を最初に呼び、`run(name, command, environment, seconds)`で実行する。既存の80℃ thermal guardと他利用者checkerを必須とし、両PDの事前/実行中state・errinfo/errtaskとpersistent latchを確認する。既存run名のlogsは上書きせずエラーにする。異常時はlatchと状態・binary・image・差分を保存して停止する。root/出力先は明示設定で、CPU/IRQ番号やユーザー名をrunner内へ固定しない。

sourceではこのdirectoryを、install後は`share/llama-bench`をPythonのimport pathへ加える。通常のbenchmarkには`environment['LLAMA_BENCH_MAIN_CPU_POLICY']='2'`等を渡す。コマンドはhelper対応binaryを直接渡し、tasksetやwrapperをargvの先頭へ追加しない。source_rootsには差分を保存する実際のworktreeを指定する。同一process内でconfigureを変更しながら並列実行しない。

## 検証用tests

`python3 -m unittest discover -s tools/llama-bench/tests -p "test_*.py" -v`でpolicyとguardのfail-closed試験を実行できる。Linuxでは`c++ -std=c++17 tools/llama-bench/tests/test-main-affinity.cpp -pthread -o <test>`でworker不変・復元の実測試験を行う。GPU kernelは起動しない。

## Gate/Up Auto基準

Gate/Up Pairは未設定または`auto`で対応graphのみ自動実行し、`1`も同じ。`0`で従来GEMVへ戻る。不正値・空文字は無効。CPU affinityは引き続きbenchmark専用の明示opt-inで、通常推論では自動固定しない。Auto版の基準識別情報は`results/gate-up-auto-20261103/final-state.json`を参照する。
