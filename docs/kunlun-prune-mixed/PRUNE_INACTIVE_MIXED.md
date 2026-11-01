# Prune inactive mixed input branches (Kunlun opt-in)

GGML_KUNLUN_PRUNE_INACTIVE_MIXED=1 specializes the input graph for pure token inference. Unset, 0 and invalid values retain the original input graph. This does not change any existing Q4/Q6/SwiGLU/Attention/GET_ROWS defaults.

Eligibility: LLAMA architecture; Q4_0 embedding stored in a Kunlun device buffer; matching input/hidden embedding dimensions; default inference context; no training, embedding-output mode, pipeline parallel, LoRA or Hadamard transforms. A specialized ubatch has token IDs and no external embeddings, mixed type vector or extra hidden-state input. All other conditions retain the original builder.

The specialized builder creates token IDs and the existing GET_ROWS only, then expands its dependencies. It does not construct inactive mixed GET_ROWS, DUP or SET_ROWS, or the external embedding placeholder. It does not clear compute flags or alter GGML forward-select semantics. Scheduler reserve/allocate/copy/wait algorithms and all device kernels remain unchanged.

The embedding graph input captures the requested 0/1 policy and whether it is token-only. The existing global topology/shape/sequence/output checks still apply. An input-form change or requested-policy change prevents reuse and follows the normal reset/build/allocate path. Token/index values can change without rebuilding; set_inputs transfers fresh data. Do not change process environment concurrently with inference; the serial same-context transition tests synchronize each call.

Actual external-embedding and mixed batches use the old builder, including necessary CPU operations. The smaller token-only initial reserve does not bypass allocation when a later batch needs the larger legacy graph. Representative transitions were tested on TinyLlama with the public extended batch API. Unknown model/adapter/pipeline configurations are not newly enabled.

Use with Q4/F32 GET_ROWS ON and new Q6_SDNN=1 / LEGACY=0 for the evaluated continuous-XPU path. Diagnosing with GGML_KUNLUN_DIAGNOSTICS=1 shows token_only, mixed and external_embd status when eligible. Existing GGML_SCHED_DEBUG=1 prints real split boundaries.

It remains opt-in. Single PD splits and graph calls decrease, but observed PP/TG changes are small or variable; Tiny TG does not show a confirmed speedup. See REPORT.ja.md for the complete accuracy, input-switch, transfer and benchmark results.
