# Kunlun K200 int8 快路径调优 patch（存档）

生成日期：2026-11-20
基线：fork `UnagiDojyou/kunlunK200_llama.cpp` @ `9e5d5a3`（vendor `bd99c9d` / 库 0.7.12）

## 这是什么

这是 K200 上让 llama.cpp 走 **int8(CDNN) 快路径** 的全套本地调优。实测在这一特定硬件组合
（K200 双 die + Pentium G4560）上，7B Q4 模型的解码速度从「上游原生 ~6 t/s」提升到 **~17 t/s**。
这些改动**从未提交到上游**，只存在于本地工作树，现整理存档。

⚠️ **状态：实验性**。包含多个诊断/探针开关，部分路径（尤其 MoE 相关）尚未验证完成。
**不要直接用于生产或服务提交。** 存档目的是留档与复现，不是提交一个已验收的功能。

## 文件说明

| 文件 | 内容 | 用途 |
|---|---|---|
| `backend-full.patch` | 主仓库全部改动（8 文件 / 32 hunk） | 完整存档，已在旧基线上验证可干净 apply |
| `vendor-full.patch` | vendor/KunlunK200 全部改动（21 文件 / 66 hunk） | 完整存档，含 int8 GEMV kernel 核心 |
| `backend-int8.patch` | 从 full 中按 hunk 索引抽取的「int8 核心」子集（5 hunk） | ⚠️ **编译不通过**，仅供分析范围参考（见下） |
| `vendor-int8.patch` | vendor 中的 int8 相关 hunk（k200.h + k200.cpp，9 hunk） | ⚠️ 同上，不可单独使用 |
| `extract_int8.py` | 抽取脚本，可从 full patch 重新生成精简版 | 索引基于本次备份，patch 变化需重新核对 |

## 重要：无法完美分离 MoE

用户要求「只交 int8，MoE 不交，其他实验不交」。技术上**做不到完全干净的分离**：

- `ggml-kunlun.cpp` 的 h02（大 hunk，+124 行）里，int8 的 `use_i8`/`I8_FORCE_*`/`KMAX_I8`
  与 MoE 的 `use_i8_expert`/`native_expert`、实验的 `self_split_on`/`fake_skip_add`
  **写在同一个 hunk 内**，按 hunk 边界无法拆开。
- `backend-int8.patch` 因此仍含 `use_i8_expert`、`native_expert`、`self_split_on` 等符号定义
  （多为未调用的死代码/实验开关）。它**不是**一个纯净的 int8-only patch。
- 若强行做行级切除，会破坏 `use_i8()` 内的引用关系。

**精简版已实测编译失败**（在基线源码应用后 `g++ -fsyntax-only` 报错）：
`supports_op` 重复定义、`supports_op_impl` 未声明、`vec_replicate_on` 未声明
——按 hunk 抽取漏掉了跨 hunk 依赖的辅助函数。因此精简 patch **仅作改动范围分析参考，不可直接使用**。

结论：只有 **full 版**是可用代码（17 t/s 就是它编出来的），但它包含全部实验代码，
无法在"只交 int8、不交 MoE 和其他"的约束下产出一个能编译的纯净子集。
这个约束在本代码库里技术上无解——int8 改动与其他实验深度交织。

## ★ 已实测验证：能跑出 17 t/s（2026-11-20）

用本目录的 **full patch + required-kernels**（下文安装步骤）在干净基线 9e5d5a3 上
重编，7B Q4_0 模型（`-sm row` + W8/int8）实测：

```
TG 16.91 / 17.11 / 17.65 t/s  → 中位数 17.11 t/s
PP ~50 t/s
```

对照：上游 0.7.14 未打 patch 的同机同配置为 6.3 t/s。**这个提交确实能把 7B 恢复到 17 t/s。**

## 如何应用（在基线 9e5d5a3 上）

⚠️ **仅 `git apply` 两个 patch 不够**——还缺 8 个 untracked 的 kernel 源文件，
否则编译报 `kunlun-dequant-q4-f32.xpu missing`。完整步骤：

```bash
cd < kunlunK200_llama.cpp 检出 >          # 基线 9e5d5a3 / vendor bd99c9d
git apply /path/to/backend-full.patch
cd vendor/KunlunK200
git apply /path/to/vendor-full.patch
cp /path/to/required-kernels/*.xpu kernels/   # ← 关键：8 个 kernel 必须补上
cd ../..
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DGGML_NATIVE=ON \
  -DKUNLUN_XRE=/usr/local/xpu-4.33.0 \
  -DKUNLUN_XPUAPI=/opt/llm/sdk/xdnn-20220919/so/libxpuapi.so \
  -DKUNLUN_XTDK=/opt/llm/sdk/xtdk-llvm15-ubuntu1604_x86_64
XDNN=/opt/llm/sdk/xdnn-20220919 cmake -S . -B build -G Ninja   # XDNN 环境变量
ninja -C build
```

运行需 W8/int8 环境变量（见 `start_qwen7b.sh`）：`GGML_KUNLUN_COMMON_Q4=0`
`GGML_KUNLUN_REQUANT_I8=1` `KUNLUN_I8_GEMM=1` + attention/flash/rope。

精简版 patch（backend-int8/vendor-int8）已实测**编译失败**，仅作改动范围参考，勿用。

## 重新生成精简版

```bash
python3 extract_int8.py backend  backend-full.patch  backend-int8.patch
python3 extract_int8.py vendor   vendor-full.patch   vendor-int8.patch
```

## 关键改动点（int8 快路径）

- `vendor/k200.h`：`WeightType::rowwise_i8`、`matmul_mt`/`i8_prepare` 声明、`kmax_w8`（K 上限放宽）
- `vendor/k200.cpp`：`upload_i8_rows`、`launch_mv`/`matmul` 的 int8 分支、`combine_mv`
- `ggml-kunlun.cpp`：`use_i8()` 判定、`KMAX_I8=65536`、`GGML_KUNLUN_I8_FORCE_*` 选层开关

## 相关背景（供接收方判断价值）

- 上游 0.7.14（提交 `2428cf6` + `f0b114a`）把 Row 路径 pair/SwiGLU 迁到 common_q4 路径，
  砍掉了 `COMMON_Q4=0`（int8）下的 MAC/DMA 加速；同机实测 7B 从 17 t/s 掉到 6.3 t/s。
- 上游新增的 tensor-split 融合特性与本 patch 是两条不同路线，可能冲突。
- 本 patch 与硬件/驱动强绑定（K200 4.33.0 驱动、xdnn-20220919 SDK、0.7.12 库 ABI）。

## 推送说明

本交付包存放于 fork `2426120557/kunlunK200_llama.cpp` 的 `wip/int8-fastpath` 分支，
仅作存档，**未提 PR**。内容经过脱敏（无凭证），可安全公开。
