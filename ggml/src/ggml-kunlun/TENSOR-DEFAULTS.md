# Kunlun K200 runtime defaults (0.7.13)

The split mode is still explicit. With `-sm tensor -dev KUNLUN0/KUNLUN1`,
the K200 backend uses the guarded two-PD executor by default for the validated
Llama 2 7B Q4_0 inference layout. This mode requires F16 K/V, FlashAttention,
a single sequence, ubatch 128, and context at most 1024. The model and context
checks reject unsupported configurations before execution.

| Mode | Default policy |
|---|---|
| Single / Layer | Use the guarded device Attention, FlashAttention, vector broadcast and existing Q4/Q6 dispatch. |
| Row | Use the cooperative Q4 path, pipelining, batched output transfer, Norm placement, Gate/Up and SwiGLU, and independent chunk progress for supported shapes. |
| Tensor | Use physical child placement and graph probe caching, grouped Attention/FFN/KV/elementwise, preflight caching, K/V max-cache row metadata, the Prefill writer, the guarded H=16 group4 path, and PD0 logits for the eligible raw Q6_K output. |
| Tensor Q4 | Use GLOBAL weight-projection quantization, fused MAX, direct down staging, WO GEMV/input sharing, and guarded WO/Down FP32 all-reduce. |

The `GGML_KUNLUN_ROW_{PARALLEL,TRANSFER,NORM,DISPATCH,PAIR,PIPELINE,GLU,OVERLAP,CHUNKS}`
controls accept explicit `0` to choose their previous fallback. For Tensor
Split, `GGML_KUNLUN_TP_EW_CHAIN=0` disables chaining with scalar operators still
eligible for two-PD execution. `TP_EW_INPLACE=0` or `TP_EW_RMS_QUEUE=0` also
disables chaining (Tensor switches have the `GGML_KUNLUN_` prefix).

`GGML_KUNLUN_TP_FA_GLOBAL_MAX=1` opts into the shared FlashAttention K/V
maxima and their numerical behavior. GLOBAL Q4 projection scales are already
the default. `GGML_KUNLUN_TP_WO_PACK_ARGS` and `GGML_KUNLUN_TP_WO_SYNC_COPY`
are opt-in. `GGML_KUNLUN_TP_WO_SUM_{ADD,COPY}` and
`GGML_KUNLUN_TP_DOWN_SUM_{ADD,COPY}` separately select the verified 4096-element
all-reduce implementation. The fast copies still require the supported K200
and XRE/driver versions. `KUNLUN_LARGE_PP=1` retains its experimental scope.

Build with the pinned `vendor/KunlunK200` submodule or point
`KUNLUN_USE_SYSTEM_LIBRARY=ON` at a compatible 0.7.13 installation.
`GGML_KUNLUN_TP_TESTS=ON` enables the optional hardware validation executables.
