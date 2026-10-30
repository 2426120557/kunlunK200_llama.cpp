# Q6_K SDNNのオンチップ集約

Q6_Kのopt-in条件とCPU fallbackは従来どおり。K2048/4096、N32000の出力層で、新libraryのEW/L1E集約を使用する。K16ごとの補正を保ち、HBM partialとCL joinを除去した。Q4、Activation lease、SwiGLU、Large PPの設定は変更していない。

| 設定 | 選択 |
|---|---|
| `KUNLUN_Q6_K_SDNN`未設定または`0` | CPU |
| `KUNLUN_Q6_K_SDNN=1`、`KUNLUN_Q6_K_LEGACY`未設定または`0` | 新オンチップSDNN |
| `KUNLUN_Q6_K_SDNN=1 KUNLUN_Q6_K_LEGACY=1` | 旧HBM partial SDNN |
| 非対応tensor/shape/stride/view | 既存fallback |

7Bの定常GEMVは4.557→2.837 ms、CPUは3.481 ms。Tinyは2.521→1.656 ms、CPUは1.693 ms。ただしTinyのホストlogits取得込みは約1.693 msでCPUとほぼ同等。Single PD0のTG128はCPU/旧/新が7B 8.473/8.400/8.530、Tiny 27.750/27.885/28.874 token/sだった（各2標本）。全shapeへの既定有効化は行わない。

新旧の実起動数を終了時counterへ反映し、新経路では`join_launches=0`を確認できる。新しい診断APIを使うため、gitlinkで指定したlibraryとそのkernelを揃えてbuildする。旧0.7.12のversion文字列だけでは新機能の有無を区別できない。

精度、guard、KV復元、Activation共有、HBM peak、再現条件は[libraryの結果](../../vendor/KunlunK200/docs/q6-onchip/REPORT.ja.md)を参照。
