bool tp_attention_enabled() {
    const char * flag=kunlun_default_on("GGML_KUNLUN_TP_ATTENTION");
    return tp_ffn_enabled() && flag && !std::strcmp(flag,"1");
}
bool tp_attention_supported(const ggml_tensor * t) {
    if(t->op==GGML_OP_ROPE){kunlun::Rope4Layout p{};return !*attention_parameters(t,p) && p.x.ne[0]==128 && p.x.ne[1]==16 && (p.x.ne[2]==1 || p.x.ne[2]==128) && p.mode==0 && p.n_offs==0 && p.n_dims==128;}
    if(t->op==GGML_OP_FLASH_ATTN_EXT){kunlun::Flash4Layout p{};return !*flash_parameters(t,p) && p.q.ne[0]==128 && p.q.ne[2]==16 && p.k.ne[2]==16 && p.q.ne[3]==1 && p.k.ne[3]==1 && p.v.ne[3]==1 && (p.q.ne[1]==1 || p.q.ne[1]==128) && p.k.ne[1]<=1024 && p.max_bias==0 && p.has_mask;}
    return false;
}
int tp_attention_execute(TPGroupExecutor * g,ggml_tensor ** t) {
    if(!tp_attention_enabled() || (t[0]->op!=GGML_OP_ROPE && t[0]->op!=GGML_OP_FLASH_ATTN_EXT))return 0;
    const bool rope=t[0]->op==GGML_OP_ROPE;
    for(int d=0;d<2;++d) {
        if(t[d]->op!=t[0]->op || !tp_attention_supported(t[d]))return -1;
        if(rope){kunlun::Rope4Layout p{};if(*attention_parameters(t[d],p) || p.x.ne[0]!=128 || p.x.ne[1]!=16)return -1;}
        else{kunlun::Flash4Layout p{};if(*flash_parameters(t[d],p) || p.q.ne[0]!=128 || p.q.ne[2]!=16 || p.k.ne[2]!=16 || p.k.ne[1]>1024)return -1;}
    }
    const char *global_max=std::getenv("GGML_KUNLUN_TP_FA_GLOBAL_MAX");
    const bool shared=!rope && global_max && !std::strcmp(global_max,"1");
    int components=3;const char *ablation=std::getenv("KUNLUN_TP_FA_MAX_COMPONENT");
    if(shared && ablation){if(!std::strcmp(ablation,"k"))components=1;else if(!std::strcmp(ablation,"v"))components=2;else return -1;}
    uint64_t begin=tp_now(),submit[2],done[2];bool ok=true;
    auto complete=[&](int d){K200_DIAG_SCOPE(complete_diag,"tp.attn","complete",t[0]->name,0,d);uint64_t wait_begin=tp_now();select(d);if(rope)kunlun::attention_rope_complete(g->contexts[d]->attention,d);else kunlun::flash_attention_complete(g->contexts[d]->attention,d);g->pending[d]=false;done[d]=tp_now();g->contexts[0]->tp_profile.attention_wait_ns[d]+=done[d]-wait_begin;};
    if(shared){
        for(int d=0;d<2;++d){K200_DIAG_SCOPE(prepare_diag,"tp.attn","prepare",t[0]->name,0,d);select(d);kunlun::Flash4Layout p{};flash_parameters(t[d],p);auto *q=t[d]->src[0],*k=t[d]->src[1],*v=t[d]->src[2],*mask=t[d]->src[3];
            g->pending[d]=kunlun::flash_attention_prepare(g->contexts[d]->attention,d,t[d]->data,ggml_nbytes(t[d]),q->data,ggml_nbytes(q),k->data,ggml_nbytes(k),v->data,ggml_nbytes(v),mask?mask->data:nullptr,mask?ggml_nbytes(mask):0,p);ok=g->pending[d] && ok;
        }
        if(ok){
            K200_DIAG_SCOPE(max_diag,"tp.attn","FA-MAX",t[0]->name);
            ok=kunlun::flash_attention_share_maxima(g->contexts[0]->attention,g->contexts[1]->attention,components);
        }
        if(!ok){for(int d=0;d<2;++d)if(g->pending[d])complete(d);return -1;}
    }
    for(int d=0;d<2;++d) {
        K200_DIAG_SCOPE(submit_diag,"tp.attn","submit",t[0]->name,0,d);
        select(d);auto * a=t[d]->src[0],*b=t[d]->src[1];
        kunlun::kv_max_invalidate(d,t[d]->data,ggml_nbytes(t[d]));
        if(rope){kunlun::Rope4Layout p{};attention_parameters(t[d],p);
            g->pending[d]=kunlun::attention_rope_submit(g->contexts[d]->attention,d,t[d]->data,ggml_nbytes(t[d]),a->data,ggml_nbytes(a),(const int32_t*)b->data,ggml_nbytes(b),p);
        }else{kunlun::Flash4Layout p{};flash_parameters(t[d],p);auto * v=t[d]->src[2],*mask=t[d]->src[3];
            g->pending[d]=shared?kunlun::flash_attention_resume(g->contexts[d]->attention,d):kunlun::flash_attention_submit(g->contexts[d]->attention,d,t[d]->data,ggml_nbytes(t[d]),a->data,ggml_nbytes(a),b->data,ggml_nbytes(b),v->data,ggml_nbytes(v),mask?mask->data:nullptr,mask?ggml_nbytes(mask):0,p);
        }
        ok=g->pending[d] && ok;submit[d]=tp_now();
        if(tp_serial() && g->pending[d])complete(d);
    }
    for(int d=0;d<2;++d)if(g->pending[d])complete(d);
    if(!ok){if(shared)for(int d=0;d<2;++d)complete(d);return -1;}
    tp_record(g,rope?TP_ROPE:TP_FLASH,begin);++g->parallel_nodes;
    if(tp_trace())GGML_LOG_INFO("KUNLUN TP %s: submit0=%llu submit1=%llu complete0=%llu complete1=%llu serial=%d\n",rope?"RoPE":"Flash",(unsigned long long)submit[0],(unsigned long long)submit[1],(unsigned long long)done[0],(unsigned long long)done[1],(int)tp_serial());
    return 1;
}

kunlun::FlashInspection tp_flash_inspection(ggml_backend_t b) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    for(int d=0;d<2;++d)if(b && ggml_backend_get_device(b)==&devices[d])return kunlun::flash_inspection(((backend_context*)b->context)->attention);
    return {};
}
