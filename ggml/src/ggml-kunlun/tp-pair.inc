int tp_pair_execute(TPGroupExecutor * g,ggml_tensor ** a,ggml_tensor ** b) {
    const char * flag=kunlun_default_on("GGML_KUNLUN_TP_PAIR");
    if(!tp_ffn_enabled() || !flag || std::strcmp(flag,"1"))return 0;
    for(int d=0;d<2;++d)if(!tp_q4_tensor(a[d],d) || !tp_q4_tensor(b[d],d) ||
        a[d]->src[0]->ne[0]!=4096 || a[d]->src[0]->ne[1]!=5504 || b[d]->src[0]->ne[1]!=5504 || a[d]->ne[1]!=1 || b[d]->ne[1]!=1 ||
        a[d]->src[1]!=b[d]->src[1] || a[d]->src[0]==b[d]->src[0])return 0;
    if(!tp_prepare_inputs(g,a))return -1;
    uint64_t begin=tp_now(),submitted_at[2];bool ok=true;
    for(int d=0;d<2;++d){K200_DIAG_SCOPE(submit_diag,"tp.stage","submit-pair",a[0]->src[0]->name,0,d);select(d);auto * p=(tensor_parts *)a[d]->src[0]->extra,*q=(tensor_parts *)b[d]->src[0]->extra;
        g->pending[d]=kunlun::common_q4_tp_submit_pair(g->contexts[d]->common[d],d,a[d]->data,b[d]->data,*p->common_weight[d],*q->common_weight[d],g->leases[d],g->invocation);
        ok=g->pending[d] && ok;submitted_at[d]=tp_now();
        if(tp_serial()){auto wait=tp_now();kunlun::common_q4_finish_submit(g->contexts[d]->common[d]);kunlun::common_q4_crop_submit(g->contexts[d]->common[d]);kunlun::common_q4_complete(g->contexts[d]->common[d]);g->pending[d]=false;g->contexts[0]->tp_profile.finish_wait_ns+=tp_now()-wait;}
    }
    tp_q4_finish(g);
    if(!ok){if(tp_trace())GGML_LOG_INFO("KUNLUN TP Pair declined; ordinary local Q4 follows\n");return 0;}
    tp_record(g,TP_GATE,begin);++g->contexts[0]->tp_profile.pair_calls;g->parallel_nodes+=2;
    if(tp_trace())GGML_LOG_INFO("KUNLUN TP Pair: local N5504 crop, no P2P, submit0=%llu submit1=%llu complete=%llu\n",
        (unsigned long long)submitted_at[0],(unsigned long long)submitted_at[1],(unsigned long long)tp_now());
    return 1;
}


bool tp_gate_glu_execute(TPGroupExecutor * g,ggml_cgraph ** graphs,
        const ggml_backend_meta_node_descriptor * desc,size_t i) {
    auto * logical=desc[i+2].logical;
    if (logical->op!=GGML_OP_GLU || logical->flags&GGML_TENSOR_FLAG_NO_FUSION ||
        (logical->ne[1]!=1 && logical->ne[1]!=128) ||
        desc[i].split_state->axis!=GGML_BACKEND_SPLIT_AXIS_0 ||
        desc[i+1].split_state->axis!=GGML_BACKEND_SPLIT_AXIS_0 ||
        desc[i+2].split_state->axis!=GGML_BACKEND_SPLIT_AXIS_0) return false;
    constexpr int width=5504, physical=5632;
    const int rows=logical->ne[1];
    ggml_tensor * mm[2][2]{}, * glu[2]{};
    const kunlun::CommonQ4Weight * weights[2][2]{};
    bool swap[2]{};
    for (int d=0;d<2;++d) {
        auto * graph=graphs[d];
        auto * first=mm[0][d]=ggml_graph_node(graph,i);
        auto * second=mm[1][d]=ggml_graph_node(graph,i+1);
        auto * out=glu[d]=ggml_graph_node(graph,i+2);
        if (out->op!=GGML_OP_GLU || *glu_reason(out) || out->view_src ||
            !tp_plain_f32(out,d) || (uintptr_t)out->data%64 || out->ne[0]!=width ||
            out->ne[1]!=rows || ggml_nrows(out)!=rows ||
            first->src[1]!=second->src[1] || first->src[0]==second->src[0] ||
            !ggml_node_has_n_uses(graph,i,1) || !ggml_node_has_n_uses(graph,i+1,1)) return false;
        swap[d]=out->src[0]==second;
        if (out->src[0]!=(swap[d]?second:first) || out->src[1]!=(swap[d]?first:second)) return false;
        for (int j=0;j<2;++j) {
            const auto * t=mm[j][d];
            if (!tp_q4_tensor(t,d) || !ggml_are_same_shape(t,out) ||
                t->src[0]->ne[0]!=4096 || t->src[0]->ne[1]!=width) return false;
            auto * p=(tensor_parts *)t->src[0]->extra;
            weights[j][d]=p->common_weight[d].get();
        }
        const auto * input=first->src[1];
        const auto y=(uintptr_t)out->data, x=(uintptr_t)input->data;
        if (y<x+ggml_nbytes(input) && x<y+ggml_nbytes(out)) return false;
    }
    if (swap[0]!=swap[1]) return false;
    const char * pair_flag=kunlun_default_on("GGML_KUNLUN_TP_PAIR");
    // Keep the same GEMV dispatch as the ordinary group, including Pair declines.
    if (rows==1 && (!pair_flag || std::strcmp(pair_flag,"1")!=0)) return false;
    const size_t bytes=(size_t)rows*physical*sizeof(float);
    void * padded[2][2]{};
    {
    K200_DIAG_SCOPE(gate_scope,"tp.node","MUL_MAT",mm[0][0]->src[0]->name);
    if (!tp_prepare_inputs(g,mm[0])) return false;
    const auto begin=tp_now();
    for (int d=0;d<2;++d) {
        auto & buf=g->gate_up[d];
        if (!buf || buf->size()<2*bytes) buf.reset(new kunlun::Buffer(d,2*bytes));
        padded[0][d]=buf->data();
        padded[1][d]=(char *)buf->data()+bytes;
    }
    const int projections=rows==1?1:2;
    for (int j=0;j<projections;++j) {
        bool ok=true;
        for (int d=0;d<2;++d) {
            auto & work=g->contexts[d]->common[d];
            K200_DIAG_SCOPE(submit_diag,"tp.stage",rows==1?"submit-pair":"submit",mm[j][0]->src[0]->name,0,d);
            g->pending[d]=rows==1 ?
                kunlun::common_q4_tp_submit_pair_padded(work,d,padded[0][d],padded[1][d],
                    *weights[0][d],*weights[1][d],g->leases[d],g->invocation) :
                kunlun::common_q4_tp_submit_padded(work,d,padded[j][d],
                    *weights[j][d],g->leases[d],g->invocation);
            ok=ok && g->pending[d];
        }
        tp_q4_finish(g);
        if (!ok) return false; // Only private scratch has been written.
    }
    tp_record(g,TP_GATE,begin);
    }
    {
        K200_DIAG_SCOPE(glu_scope,"tp.node","GLU",glu[0]->name);
        const auto start=tp_now();
        bool ok=true;
        for (int d=0;d<2;++d) {
            auto * out=glu[d];
            const kunlun::SwiGLULayout layout{width,rows,physical*4,physical*4,width*4,kunlun::InputType::f32};
            kunlun::kv_max_invalidate(d,out->data,ggml_nbytes(out));
            g->pending[d]=kunlun::swiglu_submit(g->contexts[d]->glu,d,out->data,ggml_nbytes(out),
                    padded[swap[d]][d],bytes,padded[!swap[d]][d],bytes,layout);
            ok=ok && g->pending[d];
        }
        if (!g->drain()) GGML_ABORT("KUNLUN TP Gate/Up SwiGLU drain failed");
        if (!ok) return false;
        tp_record(g,TP_GLU,start);
    }
    g->parallel_nodes+=3;
    if (rows==1) ++g->contexts[0]->tp_profile.pair_calls;
    ++g->gate_glu_calls[rows==1?0:1];
    return true;
}
