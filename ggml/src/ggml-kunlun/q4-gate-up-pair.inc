bool gate_up_enabled() {
    const char *s=std::getenv("GGML_KUNLUN_Q4_GATE_UP_PAIR");
    return common_enabled() && (!s || !std::strcmp(s,"auto") || !std::strcmp(s,"1"));
}
bool gate_up_tensor(backend_context *c,const ggml_tensor *t) {
    if(!activation_candidate(c,t) || t->type!=GGML_TYPE_F32 || t->view_src || t->src[1]->view_src ||
       !ggml_is_contiguous(t) || ggml_nrows(t)!=1 || t->src[0]->type!=GGML_TYPE_Q4_0)return false;
    auto *b=buffer_of(t),*w=buffer_of(t->src[0]);auto *parts=(tensor_parts*)t->src[0]->extra;
    if(!b || b->type->split || b->type->id!=c->id || !w || w->type->split || w->type->id!=c->id ||
       !parts || !parts->common_weight[c->id])return false;
    auto a=parts->common_weight[c->id]->info();
    return a.k==a.kp && a.n==a.np && ((a.k==4096 && a.n==11008) || (a.k==2048 && a.n==5632));
}
struct GateUpGraphScope {
    backend_context *c;
    std::map<const ggml_tensor*,ggml_tensor*> plan;
    GateUpGraphScope(backend_context *owner,ggml_cgraph *graph):c(owner) {
        const bool enabled=gate_up_enabled();
        if(c->pair_diagnostics)c->pair_modes_seen|=enabled?2:1;
        if(!enabled)return;
        auto skip=[&](size_t reason){if(c->pair_diagnostics)++c->pair_skips[reason];};
        std::map<const ggml_tensor*,int> uses;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)for(auto *x:ggml_graph_node(graph,i)->src)if(x)++uses[x];
        for(int i=0;i+2<ggml_graph_n_nodes(graph);++i){
            auto *a=ggml_graph_node(graph,i),*b=ggml_graph_node(graph,i+1),*glu=ggml_graph_node(graph,i+2);
            if(a->op!=GGML_OP_MUL_MAT || b->op!=GGML_OP_MUL_MAT)continue;
            if(!gate_up_tensor(c,a) || !gate_up_tensor(c,b)){skip(0);continue;}
            if(a->src[1]!=b->src[1]){skip(1);continue;}
            if(uses[a]!=1 || uses[b]!=1){skip(2);continue;}
            if(glu->op!=GGML_OP_GLU || ggml_get_glu_op(glu)!=GGML_GLU_OP_SWIGLU ||
               !((glu->src[0]==a && glu->src[1]==b)||(glu->src[0]==b && glu->src[1]==a))){skip(3);continue;}
            if(!ggml_are_same_shape(a,b) || a->src[0]==b->src[0]){skip(4);continue;}
            plan.emplace(a,b);++c->pair_candidates;
        }
        if(plan.empty())skip(5);
    }
    bool decline(size_t reason) {
        ++c->pair_fallbacks;
        if(c->pair_diagnostics && reason<9)++c->pair_reasons[reason];
        return false;
    }
    bool execute(ggml_tensor *a) {
        auto at=plan.find(a);if(at==plan.end())return false;
        auto *b=at->second,*x=a->src[1];int id=c->id,k=(int)x->ne[0];
        auto *wa=((tensor_parts*)a->src[0]->extra)->common_weight[id].get();
        auto *wb=((tensor_parts*)b->src[0]->extra)->common_weight[id].get();
        // Both tensors are adjacent and have no consumer before this pair completes.
        activation_write(c,a);activation_write(c,b);
        kunlun::CommonQ4ActivationLease local;
        kunlun::CommonQ4ActivationLease *lease=&local;
        uint64_t generation=0;
        auto &share=c->activation;
        if(share.enabled){
            activation_begin(c,a);if(!share.use)return decline(6);
            lease=&share.lease[id];generation=share.generation;
        }else{GGML_ASSERT(++c->pair_generation!=0);generation=c->pair_generation;}
        if(!lease->matches(id,k,1,kunlun::InputType::f32,generation)){
            if(!kunlun::common_q4_prepare(c->common[id],id,x->data,k,1,kunlun::InputType::f32,generation,*lease)){
                share.use=false;return decline(6);
            }
            if(share.enabled)++share.prepares[id];
        }
        kunlun::kv_max_invalidate(id,a->data,ggml_nbytes(a));kunlun::kv_max_invalidate(id,b->data,ggml_nbytes(b));
        if(!kunlun::common_q4_matmul_pair(c->common[id],id,a->data,b->data,*wa,*wb,*lease,generation)){
            share.use=false;
            return decline(c->pair_diagnostics?(size_t)kunlun::common_q4_pair_last_status(c->common[id]):0);
        }
        if(share.enabled){activation_end(c,a);activation_end(c,b);}
        common_paths[id][0]+=2;common_m[id][1]+=2;
        const char *phase=std::getenv("K200_PROFILE_PHASE");common_phase[id][phase?phase:"other"][0]+=2;
        ++c->pair_calls;return true;
    }
};
