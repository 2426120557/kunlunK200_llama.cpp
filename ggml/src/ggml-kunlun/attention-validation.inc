static void packed_attention(LayerRun & run,LayerData & data) {
    using Query=kunlun::CommonQ4Info (*)(const ggml_tensor *);auto query=(Query)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_weight_info");REQUIRE(query);
    for(int op:{WQ,WK,WV,WO})for(int d=0;d<2;++d){auto * t=ggml_backend_meta_tensor_child(run.weights[0][op],d);auto a=query(t);REQUIRE(a.device==d && a.abi_version==1 && a.layout==1 && a.scale_order==1 && a.k==t->ne[0] && a.n==t->ne[1]);
        std::vector<unsigned char> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());const auto & original=data.layers[0][op].bytes;
        if(op!=WO)REQUIRE(std::equal(raw.begin(),raw.end(),original.begin()+d*raw.size()));
        else for(int row=0;row<4096;++row)REQUIRE(std::memcmp(raw.data()+(size_t)row*1152,original.data()+(size_t)row*2304+d*1152,1152)==0);
        auto packed=device_read<unsigned char>(d,t->data,a.payload_bytes);
        for(int col=0;col<a.np;++col)for(int kb=0;kb<a.kp/32;++kb){bool valid=col<a.n && kb<a.k/32;size_t si=a.scale_offset+((size_t)kb*a.np+col)*2,src=((size_t)col*(a.k/32)+kb)*18;REQUIRE(packed[si]==(valid?raw[src]:0) && packed[si+1]==(valid?raw[src+1]:0));
            auto q=[&](int j){return valid?(((raw[src+2+j%16]>>(j/16*4))&15)-8)&15:0;};
            for(int p=0;p<16;++p){int row=(col%32/16)*8+p/2,bank=col%16;size_t wi=((size_t)(col/32)*(a.kp/32)+kb)*512+(bank*16+row)*2+p%2;REQUIRE(packed[wi]==(q(p*2)|(q(p*2+1)<<4)));}
        }
        std::printf("PASS Attention packed scale/nibble %s PD%d K=%d N=%d\n",t->name,d,a.k,a.n);
    }
}
static void wo_quantization(LayerGraph & graph,LayerData & data,LayerGraph & reference) {
    using Query=kunlun::CommonQ4Quantization (*)(ggml_backend_t);auto query=(Query)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_quantization");REQUIRE(query);
    QuantCopy local[2];for(int d=0;d<2;++d)local[d]=snapshot(d,query(ggml_backend_meta_simple_backend(graph.r.backend,d)));
    auto input=read(graph.stages[0].at("attn_flat"));kunlun::Buffer full_input(0,input.size()*4);device_write(0,full_input.data(),input.data(),input.size()*4);
    kunlun::CommonQ4Workspace full_work;kunlun::CommonQ4ActivationLease full_lease;REQUIRE(kunlun::common_q4_prepare(full_work,0,full_input.data(),4096,graph.m,kunlun::InputType::f32,7001,full_lease));auto full=snapshot(0,full_work.quantization());
    for(int d=0;d<2;++d){auto & a=local[d];REQUIRE(a.kp==2048 && full.kp==4096 && a.m==full.m && a.components==full.components && a.params==full.params);
        if(a.components==2){for(int component=0;component<2;++component)for(int j=0;j<2048;++j){size_t dst=component*a.kp*2+(j/32)*64+j%32,global=d*2048+j;REQUIRE(a.q[dst]==full.q[component*full.kp*2+(global/32)*64+global%32]);}
            for(int component=0;component<2;++component)for(int b=0;b<a.kp/32;++b)for(int j=32;j<64;++j)REQUIRE(a.q[component*a.kp*2+b*64+j]==0);
            for(int j=4*a.kp;j<4*a.kp+1024;++j)REQUIRE(a.q[j]==0);
        }else for(int row=0;row<a.m;++row)for(int j=0;j<2048;++j)REQUIRE(a.q[row*a.kp+j]==full.q[row*4096+d*2048+j]);
    }
    std::printf("PASS WO GLOBAL byte/scale bitwise vs same full-K input M=%d\n",graph.m);
    const size_t xs=(size_t)2048*graph.m*4,ys=(size_t)4096*graph.m*4;kunlun::Buffer x0(0,xs+512),x1(1,xs+512),y0(0,ys+512),y1(1,ys+512);
    void * x[2]={(char*)x0.data()+256,(char*)x1.data()+256},*y[2]={(char*)y0.data()+256,(char*)y1.data()+256};
    std::vector<float> sources[2]={std::vector<float>(xs/4+128,7777),std::vector<float>(xs/4+128,7777)},guard(ys/4+128,-8888);
    for(int d=0;d<2;++d){for(int row=0;row<graph.m;++row)std::memcpy(sources[d].data()+64+row*2048,input.data()+row*4096+d*2048,xs/graph.m);device_write(d,d?x1.data():x0.data(),sources[d].data(),xs+512);device_write(d,d?y1.data():y0.data(),guard.data(),ys+512);}
    kunlun::CommonQ4Workspace w0,w1;kunlun::CommonQ4ActivationLease l0,l1;REQUIRE(kunlun::common_q4_tp_prepare_global(w0,w1,x[0],x[1],2048,graph.m,7002,l0,l1));
    using Handle=const kunlun::CommonQ4Weight *(*)(const ggml_tensor*);auto handle=(Handle)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_weight_handle");REQUIRE(handle);
    kunlun::CommonQ4Workspace * work[2]={&w0,&w1};kunlun::CommonQ4ActivationLease * leases[2]={&l0,&l1};
    for(int d=0;d<2;++d)REQUIRE(kunlun::common_q4_tp_submit(*work[d],d,y[d],*handle(ggml_backend_meta_tensor_child(graph.r.weights[0][WO],d)),*leases[d],7002));
    for(int d=0;d<2;++d)kunlun::common_q4_finish_submit(*work[d]);for(int d=0;d<2;++d)kunlun::common_q4_complete(*work[d]);
    std::vector<float> partial[2];
    for(int d=0;d<2;++d){auto actual=device_read<float>(d,d?y1.data():y0.data(),ys/4+128);for(int j=0;j<64;++j)REQUIRE(actual[j]==-8888 && actual[ys/4+64+j]==-8888);partial[d]=std::vector<float>(actual.begin()+64,actual.end()-64);REQUIRE(device_read<float>(d,d?x1.data():x0.data(),xs/4+128)==sources[d]);
        if(graph.m==1){auto * wt=ggml_backend_meta_tensor_child(graph.r.weights[0][WO],d);std::vector<unsigned char> raw(ggml_nbytes(wt));ggml_backend_tensor_get(wt,raw.data(),0,raw.size());compare("WO partial vs CPU Q4/F64",partial[d],cpu_matrix(raw,2048,4096,std::vector<float>(sources[d].begin()+64,sources[d].end()-64)),0.001);}
    }
    for(size_t i=0;i<partial[0].size();++i)partial[0][i]+=partial[1][i];compare("WO SUM vs CPU P0+P1",result(graph.r,graph.stages[0].at("wo")),partial[0],1e-6);
    kunlun::Buffer same_input_output(0,ys);
    REQUIRE(kunlun::common_q4_matmul_shared(full_work,0,same_input_output.data(),*handle(reference.r.weights[0][WO]),full_lease,7001));
    compare("WO vs Single same Attention input",result(graph.r,graph.stages[0].at("wo")),device_read<float>(0,same_input_output.data(),ys/4),1e-5);
    if(graph.m==1)compare("WO vs CPU Q4/F64 same Attention input",result(graph.r,graph.stages[0].at("wo")),cpu_matrix(data.layers[0][WO].bytes,4096,4096,input),0.001);
    std::puts("PASS WO partial guard / same-input arithmetic / SUM / quantization isolation");
}
