# Q4 GEMV address preparation opt-in

This uses the separately built Kunlun library feature `KUNLUN_Q4_DMA_ADDR32=1`.
Unset/0 keeps the existing GEMV kernels. The public ABI and llama.cpp execution
code are unchanged. Use the matching candidate library and its kernel directory;
the base submodule pin is 4413940 plus the separately supplied library patch.

The opt-in retains stream/N-major selection, common Q4 ABI v1, activation sharing,
GEMM, FlashAttention, SwiGLU, Q6 on-chip GEMV, GET_ROWS, and mixed graph pruning.
It computes bounded 32-bit offsets before the same wide global-pointer addition.
No fence or barrier is removed. See the library `docs/Q4_GEMV_DMA_ADDR32.md` for
supported shapes and `results/gemv-dmain1-20261102/REPORT.ja.md` in the development
workspace for measurements, limits, and reproduction commands. No default changed.
