# Q4 Gate/Up pair (opt-in)

`GGML_KUNLUN_Q4_GATE_UP_PAIR=1` enables the experimental pair path. Unset, `0`, and invalid values preserve ordinary GEMV/GEMM. Evaluate with `KUNLUN_Q4_GEMV_SUM_FAST=0` and `GGML_KUNLUN_SWIGLU_ABSMAX=0`.

The backend requires two adjacent F32 M1 common-Q4 matmuls with the same input tensor, followed immediately by a SwiGLU using both outputs. Both outputs must have one graph consumer. Only K4096/N11008 and K2048/N5632 are supported. Views, split buffers, output aliases, other dtypes/shapes and graph partitions fall back. No plan or lease crosses graph execution boundaries.

The library takes two existing ABI-v1 weight handles, separate output addresses, and one completed activation lease. No weights are concatenated, copied, or repacked. The caller holds `runtime_mutex` through preparation and execution. All library entry points also take that recursive lock.

One SDNN invocation preloads the quantized activation into L1D once per resident window. It runs the unchanged eight-owner pipeline for the first weight, drains all engine fences and synchronizes controllers, then runs the same pipeline for the second weight. L1D is read-only between projections; weight shuffle writes L1W and scale/correction uses the existing EW buffers. Existing inner-loop DMA/MAC overlap, K32 scale correction and partial ordering remain intact.

L3 contains two non-overlapping partial arrays. Requests are 11,272,192 bytes for 7B or 2,883,584 bytes for Tiny. The existing shared NativeL3 pool is reserved under the global lock. No other op can reuse it until the pair sum has completed. Failure to grow the pool returns false before output writes, and ordinary GEMV may reserve the smaller single-output workspace.

One CL grid sums both arrays independently, using the old ordered FP32 K32 loop. Each output is a separate F32 tensor. The pair API requires output DMA alignment (64 bytes), rejects input/output/weight overlap and stale or foreign leases, and completes both outputs before returning. A false return writes no output.

With activation sharing enabled, the graph path uses the existing lease and releases exactly two consumers after success. With sharing disabled, it owns one local lease for the pair. On success the graph loop skips the second matmul once. On rejection neither node is skipped. Other consumers of the same activation can continue using the existing lease; extra consumers of either output disable pairing.

Normal SDNN GEMV/GEMM sources and images are unchanged. New images are `kunlun-common-gemv-pair-nmajor`, `kunlun-common-gemv-pair-stream`, and `kunlun-common-sum-pair`. Host counters distinguish logical matmuls from physical SDNN launches. `common_q4_pair_statistics` reports successful pairs, declines and requested L3 partial bytes.

Performance must be checked in both prewarmed same-context comparisons and ordinary independent processes. The known timing bimodality is not resolved by this change. Keep the default OFF and consult `results/gate-up-pair-20261103/REPORT.ja.md` before using the opt-in.

Runtime defaults updated 2026-11-04: see RUNTIME_DEFAULTS.md.
