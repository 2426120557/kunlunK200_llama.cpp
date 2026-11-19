uint64_t tp_now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
void tp_record(TPGroupExecutor * g,ggml_kunlun_tp_phase phase,uint64_t begin){auto & p=g->contexts[0]->tp_profile;p.ns[phase]+=tp_now()-begin;++p.calls[phase];}
bool tp_serial() { const char * s=std::getenv("GGML_KUNLUN_TP_SERIAL");return s && !std::strcmp(s,"1"); }
bool tp_ffn_enabled() {
    const char * s=kunlun_default_on("GGML_KUNLUN_TP_FFN");return tp_enabled() && s && !std::strcmp(s,"1");
}
bool tp_q4_binding(const ggml_tensor * w,int d) {
    auto * b=buffer_of(w);auto * p=(tensor_parts *)w->extra;
    if(w->type!=GGML_TYPE_Q4_0 || w->view_src || !b || b->type->split ||
       !p || !p->common || !p->common_weight[d] || p->common_weight[d]->data()!=w->data)return false;
    auto info=p->common_weight[d]->info();uintptr_t base=(uintptr_t)b->base,ptr=(uintptr_t)w->data;
    return info.k==w->ne[0] && info.n==w->ne[1] && info.device==d && ptr>=base && ptr-base<=b->size &&
        info.payload_bytes<=b->size-(ptr-base);
}
bool tp_q4_tensor(const ggml_tensor * t,int d) {
    if(t->op!=GGML_OP_MUL_MAT || !tp_plain_f32(t,d) || !tp_plain_f32(t->src[1],d))return false;
    const auto * w=t->src[0];
    if(!tp_span(w,d) || !tp_q4_binding(w,d))return false;
    const char * flag=kunlun_default_on("GGML_KUNLUN_TP_ATTENTION");
    bool attention=flag && !std::strcmp(flag,"1") && ((w->ne[0]==4096 && w->ne[1]==2048) || (w->ne[0]==2048 && w->ne[1]==4096));
    return (t->ne[1]==1 || t->ne[1]==128) &&
        (attention || (w->ne[0]==4096 && w->ne[1]==5504) || (w->ne[0]==5504 && w->ne[1]==4096));
}
void tp_q4_finish(TPGroupExecutor * g) {
    uint64_t begin=tp_now();
    for(int d=0;d<2;++d){K200_DIAG_SCOPE(stage_diag,"tp.stage","finish","",0,d);select(d);kunlun::common_q4_finish_submit(g->contexts[d]->common[d]);}
    for(int d=0;d<2;++d){K200_DIAG_SCOPE(stage_diag,"tp.stage","crop","",0,d);select(d);kunlun::common_q4_crop_submit(g->contexts[d]->common[d]);}
    for(int d=0;d<2;++d){K200_DIAG_SCOPE(stage_diag,"tp.stage","complete","",0,d);select(d);kunlun::common_q4_complete(g->contexts[d]->common[d]);g->pending[d]=false;}
    g->contexts[0]->tp_profile.finish_wait_ns+=tp_now()-begin;
}
bool tp_prepare_inputs(TPGroupExecutor * g,ggml_tensor ** t) {
    const int k=t[0]->src[0]->ne[0],m=t[0]->ne[1];
    const char * quant=std::getenv("GGML_KUNLUN_TP_QUANT");
    if(quant && std::strcmp(quant,"global") && std::strcmp(quant,"local"))return false;
    uint64_t begin=tp_now();bool fresh=false;
    K200_DIAG_SCOPE(prepare_diag,"tp.stage","prepare",t[0]->src[0]->name);
    bool ready=true;
    bool global=(k==5504 || k==2048) && (!quant || !std::strcmp(quant,"global"));
    if(global) {
        K200_DIAG_SCOPE(global_diag,"tp.stage","global-prepare",t[0]->src[0]->name);
#ifdef K200_COMPUTE_HOOKS
        uint64_t max_before=0;
        if(k200_diag_event)max_before=g->contexts[0]->common[0].tp_statistics().max_ns;
#endif
        fresh=true;
        ready=kunlun::common_q4_tp_prepare_global(g->contexts[0]->common[0],g->contexts[1]->common[1],
            t[0]->src[1]->data,t[1]->src[1]->data,k,m,g->invocation,g->leases[0],g->leases[1],!tp_serial());
#ifdef K200_COMPUTE_HOOKS
        if(k200_diag_event){
            const auto delta=g->contexts[0]->common[0].tp_statistics().max_ns-max_before;
            k200_diag_event("tp.counter","q4-max",t[0]->src[0]->name,0,1,0,delta,0,0);
        }
#endif
        for(int d=0;d<2;++d)g->inputs[d]=t[d]->src[1];
    }
    if(!global)for(int d=0;d<2;++d) {
        auto & work=g->contexts[d]->common[d];auto & lease=g->leases[d];auto * x=t[d]->src[1];
        if(g->inputs[d]!=x || !lease.matches(d,k,m,kunlun::InputType::f32,g->invocation)) {
            fresh=true;select(d);
            bool ok=kunlun::common_q4_prepare_submit(work,d,x->data,k,m,g->invocation,lease);
            if(!ok)ok=kunlun::common_q4_prepare(work,d,x->data,k,m,kunlun::InputType::f32,g->invocation,lease);
            ready=ok && ready;g->inputs[d]=x;
            if(tp_serial()){select(d);kunlun::common_q4_prepare_complete(work);}
        }
    }
    for(int d=0;d<2;++d){select(d);kunlun::common_q4_prepare_complete(g->contexts[d]->common[d]);}
    if(!ready)return false;
    if(fresh)tp_record(g,k==2048?TP_WO_QUANT:(k==5504?TP_DOWN_QUANT:TP_GATE_QUANT),begin);
    return true;
}
int tp_q4_execute(TPGroupExecutor * g,ggml_tensor ** t) {
    if(!tp_ffn_enabled() || (t[0]->op!=GGML_OP_MUL_MAT && t[1]->op!=GGML_OP_MUL_MAT))return 0;
    if(t[0]->src[0]->type!=GGML_TYPE_Q4_0 && t[1]->src[0]->type!=GGML_TYPE_Q4_0)return 0;
    if(!tp_q4_tensor(t[0],0) || !tp_q4_tensor(t[1],1))return -1;
    if(!tp_prepare_inputs(g,t))return -1;
    const int k=t[0]->src[0]->ne[0],m=t[0]->ne[1];
    const uint64_t begin=tp_now();
    bool submitted=true;int64_t times[2];
    for(int d=0;d<2;++d) {
        K200_DIAG_SCOPE(submit_diag,"tp.stage","submit",t[0]->src[0]->name,0,d);
        select(d);auto * p=(tensor_parts *)t[d]->src[0]->extra;
        kunlun::kv_max_invalidate(d,t[d]->data,ggml_nbytes(t[d]));
        bool ok=kunlun::common_q4_tp_submit(g->contexts[d]->common[d],d,t[d]->data,*p->common_weight[d],g->leases[d],g->invocation);
        g->pending[d]=ok;submitted=ok && submitted;
        times[d]=tp_now();
        if(tp_serial()) {uint64_t wait=tp_now();
            auto & work=g->contexts[d]->common[d];select(d);kunlun::common_q4_finish_submit(work);
            kunlun::common_q4_crop_submit(work);kunlun::common_q4_complete(work);g->pending[d]=false;g->contexts[0]->tp_profile.finish_wait_ns+=tp_now()-wait;
        }
    }
    tp_q4_finish(g);
    if(!submitted)return -1;
    ++g->parallel_nodes;
    tp_record(g,k==2048?TP_WO:(t[0]->src[0]->ne[1]==2048?TP_QKV:(k==5504?TP_DOWN:(std::strstr(t[0]->src[0]->name,"ffn_up")?TP_UP:TP_GATE))),begin);
    if(tp_trace())GGML_LOG_INFO("KUNLUN TP Q4: K=%d M=%d submit0=%lld submit1=%lld complete_both=1 prepared_reuse=1\n",k,m,(long long)times[0],(long long)times[1]);
    return 1;
}
int tp_glu_execute(TPGroupExecutor * g,ggml_tensor ** t) {
    if(!tp_ffn_enabled() || (t[0]->op!=GGML_OP_GLU && t[1]->op!=GGML_OP_GLU))return 0;
    for(int d=0;d<2;++d)if(t[d]->op!=GGML_OP_GLU || *glu_reason(t[d]) || !t[d]->src[1] ||
        t[d]->type!=GGML_TYPE_F32 || t[d]->ne[0]!=5504 || (t[d]->ne[1]!=1 && t[d]->ne[1]!=128))return -1;
    uint64_t begin=tp_now();bool ok=true;
    for(int d=0;d<2;++d){auto * a=t[d]->src[0],*b=t[d]->src[1];
        kunlun::SwiGLULayout p{5504,(int)t[d]->ne[1],(int)a->nb[1],(int)b->nb[1],(int)t[d]->nb[1],kunlun::InputType::f32};
        select(d);g->pending[d]=kunlun::swiglu_submit(g->contexts[d]->glu,d,t[d]->data,ggml_nbytes(t[d]),a->data,ggml_nbytes(a),b->data,ggml_nbytes(b),p);
        ok=g->pending[d] && ok;
        if(tp_serial()){select(d);check(kunlun::wait());g->pending[d]=false;}
    }
    if(!g->drain())GGML_ABORT("KUNLUN TP GLU drain failed");
    if(ok)tp_record(g,TP_GLU,begin);
    return ok?1:-1;
}

kunlun::CommonQ4Quantization tp_quantization(ggml_backend_t b) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    for(int d=0;d<2;++d)if(b && ggml_backend_get_device(b)==&devices[d])return ((backend_context *)b->context)->common[d].quantization();
    return {};
}
kunlun::CommonQ4TPStatistics tp_quant_statistics(ggml_backend_t b) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    for(int d=0;d<2;++d)if(b && ggml_backend_get_device(b)==&devices[d])return ((backend_context *)b->context)->common[d].tp_statistics();
    return {};
}
kunlun::CommonQ4Info tp_weight_info(const ggml_tensor * t) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    auto * b=t?buffer_of(t):nullptr;auto * p=t?(tensor_parts *)t->extra:nullptr;
    if(t && t->type==GGML_TYPE_Q4_0 && !t->view_src && b && !b->type->split && p && p->common && p->common_weight[b->type->id])return p->common_weight[b->type->id]->info();
    return {};
}

const kunlun::CommonQ4Weight * tp_weight_handle(const ggml_tensor * t) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    auto * b=t?buffer_of(t):nullptr;auto * p=t?(tensor_parts *)t->extra:nullptr;
    if(t && t->type==GGML_TYPE_Q4_0 && !t->view_src && b && !b->type->split && p && p->common)return p->common_weight[b->type->id].get();
    return nullptr;
}
kunlun::CommonQ4Statistics tp_common_statistics(ggml_backend_t b) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    for(int d=0;d<2;++d)if(b && ggml_backend_get_device(b)==&devices[d])return ((backend_context *)b->context)->common[d].statistics();
    return {};
}

#include "tp-pair.inc"
bool tp_allreduce_profile(void * opaque,ggml_tensor ** tensors){auto * g=(TPGroupExecutor *)opaque;auto begin=tp_now();bool ok=tp_allreduce(opaque,tensors);if(ok)tp_record(g,TP_SUM,begin);return ok;}
bool tp_allreduce_ready(void * opaque,ggml_tensor ** t) {
    auto * g=(TPGroupExecutor *)opaque;
    const auto begin=tp_now();
    const char * flag=kunlun_default_on("GGML_KUNLUN_TP_SUM_READY");
    const bool complete=flag && std::strcmp(flag,"1")==0;
    const bool ok=tp_allreduce_impl(opaque,t,complete);
    if (ok && tp_ffn_enabled()) tp_record(g,TP_SUM,begin);
    return ok;
}
ggml_kunlun_tp_profile tp_profile_query(ggml_backend_t b){std::lock_guard<std::recursive_mutex> lock(runtime_mutex);for(int d=0;d<2;++d)if(b && ggml_backend_get_device(b)==&devices[d])return ((backend_context *)b->context)->tp_profile;return {};}

kunlun::CommonQ4ActivationStatistics tp_activation_statistics(ggml_backend_t b) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    for(int d=0;d<2;++d)if(b && ggml_backend_get_device(b)==&devices[d])return ((backend_context *)b->context)->common[d].activation_statistics();
    return {};
}

#include "tp-attention.inc"
