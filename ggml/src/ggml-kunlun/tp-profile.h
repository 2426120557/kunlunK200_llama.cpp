#pragma once
#include <cstdint>
enum ggml_kunlun_tp_phase { TP_GATE_QUANT,TP_GATE,TP_UP,TP_GLU,TP_DOWN_QUANT,TP_DOWN,TP_SUM,TP_SYNC,TP_QKV,TP_WO_QUANT,TP_WO,TP_ROPE,TP_FLASH,TP_KV,TP_NORM,TP_PHASE_COUNT };
struct ggml_kunlun_tp_profile {
    uint64_t ns[TP_PHASE_COUNT]={},calls[TP_PHASE_COUNT]={};
    uint64_t finish_wait_ns=0,pair_calls=0,attention_wait_ns[2]={};
};
