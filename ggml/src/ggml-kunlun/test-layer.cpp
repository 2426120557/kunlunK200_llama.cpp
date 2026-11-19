#define KUNLUN_FFN_TEST_NO_MAIN
#include "test-ffn.cpp"
#include <string>
#include <map>
#include <algorithm>

enum { WQ,WK,WV,WO,AN,WG,WU,WD,FN,NW };
struct LayerData {
    ggml_context * info=nullptr;gguf_context * file=nullptr;
    struct Raw {std::string name;ggml_type type;int64_t k,n;std::vector<unsigned char> bytes;};
    std::vector<std::array<Raw,NW>> layers;
    LayerData(const char * path,int count):layers(count){
        file=gguf_init_from_file(path,{true,&info});REQUIRE(file && info);std::ifstream in(path,std::ios::binary);
        const char * suffix[NW]={"attn_q","attn_k","attn_v","attn_output","attn_norm","ffn_gate","ffn_up","ffn_down","ffn_norm"};
        for(int l=0;l<count;++l)for(int j=0;j<NW;++j){auto & a=layers[l][j];a.name="blk."+std::to_string(l)+"."+suffix[j]+".weight";auto * t=ggml_get_tensor(info,a.name.c_str());REQUIRE(t);a.type=t->type;a.k=t->ne[0];a.n=t->ne[1];REQUIRE(a.type==(j==AN || j==FN?GGML_TYPE_F32:GGML_TYPE_Q4_0));
            a.bytes.resize(ggml_nbytes(t));int id=gguf_find_tensor(file,a.name.c_str());REQUIRE(id>=0);in.seekg(gguf_get_data_offset(file)+gguf_get_tensor_offset(file,id));in.read((char*)a.bytes.data(),a.bytes.size());REQUIRE((size_t)in.gcount()==a.bytes.size());}
    }
    ~LayerData(){ggml_free(info);gguf_free(file);}
};
struct LayerRun {
    bool meta;int capacity=768;ggml_backend_t backend,cpu=nullptr;
    ggml_context * wc=ctx(),*kc=nullptr;ggml_backend_buffer_t wb=nullptr,kb=nullptr;
    std::vector<std::array<ggml_tensor*,NW>> weights;std::vector<ggml_tensor*> k,v;ggml_tensor * guards[2]={};
    std::unique_ptr<llama_model> model;llama_meta_device_get_split_state_userdata policy{};
    LayerRun(LayerData & data,bool split):meta(split),weights(data.layers.size()),k(weights.size()),v(weights.size()){
        for(size_t l=0;l<weights.size();++l)for(int j=0;j<NW;++j){auto & a=data.layers[l][j];weights[l][j]=ggml_new_tensor_2d(wc,a.type,a.k,a.n);ggml_set_name(weights[l][j],a.name.c_str());}
        if(meta){model.reset(llama_model_create(LLM_ARCH_LLAMA,llama_model_default_params()));REQUIRE(model);auto & h=model->hparams;h.n_embd=4096;h.n_layer_all=32;h.n_ff_arr.fill(11008);h.n_head_arr.fill(32);h.n_head_kv_arr.fill(32);h.n_embd_head_k_full=128;h.n_embd_head_v_full=128;h.n_rot_full=128;
            for(auto & w:weights)for(auto * t:w)model->tensors_by_name.emplace_back(t->name,t);
            policy={2,model.get()};auto reg=ggml_backend_kunlun_reg();ggml_backend_dev_t ds[2]={ggml_backend_reg_dev_get(reg,0),ggml_backend_reg_dev_get(reg,1)};
            backend=ggml_backend_dev_init(ggml_backend_meta_device(ds,2,llama_meta_device_get_split_state,&policy),nullptr);cpu=ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),nullptr);
        }else backend=ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_kunlun_reg(),0),nullptr);
        REQUIRE(backend);wb=ggml_backend_alloc_ctx_tensors(wc,backend);REQUIRE(wb);
        for(size_t l=0;l<weights.size();++l)for(int j=0;j<NW;++j){auto & raw=data.layers[l][j].bytes;auto * t=weights[l][j];ggml_backend_tensor_set(t,raw.data(),0,raw.size());std::vector<unsigned char> check(raw.size());ggml_backend_tensor_get(t,check.data(),0,check.size());REQUIRE(check==raw);}
        reset_cache();std::printf("PASS real weights roundtrip %s layers=%zu\n",meta?"Meta":"Single",weights.size());
    }
    void reset_cache(){
        if(kb){ggml_backend_synchronize(backend);ggml_backend_buffer_free(kb);ggml_free(kc);}
        kc=ctx();guards[0]=ggml_new_tensor_1d(kc,GGML_TYPE_I32,256);
        for(size_t l=0;l<weights.size();++l){k[l]=ggml_new_tensor_2d(kc,GGML_TYPE_F16,4096,capacity);v[l]=ggml_new_tensor_2d(kc,GGML_TYPE_F16,4096,capacity);ggml_format_name(k[l],"cache_k_l%zu",l);ggml_format_name(v[l],"cache_v_l%zu",l);}
        guards[1]=ggml_new_tensor_1d(kc,GGML_TYPE_I32,256);kb=ggml_backend_alloc_ctx_tensors(kc,backend);REQUIRE(kb);ggml_backend_buffer_clear(kb,0);
        std::vector<unsigned> pattern(256,0x1234abcd);for(auto * t:guards)ggml_backend_tensor_set(t,pattern.data(),0,1024);
        if(meta)for(size_t l=0;l<weights.size();++l)for(auto * t:{k[l],v[l]})for(int d=0;d<2;++d){auto * child=ggml_backend_meta_tensor_child(t,d);REQUIRE(child && child->ne[0]==2048 && child->ne[1]==capacity && ggml_is_contiguous(child));}
    }
    void check_guards(){for(auto * t:guards)for(int d=0;d<(meta?2:1);++d){auto * p=meta?ggml_backend_meta_tensor_child(t,d):t;std::vector<unsigned> a(256);ggml_backend_tensor_get(p,a.data(),0,1024);REQUIRE(a==std::vector<unsigned>(256,0x1234abcd));}}
    ~LayerRun(){ggml_backend_synchronize(backend);ggml_backend_buffer_free(kb);ggml_free(kc);ggml_backend_buffer_free(wb);ggml_free(wc);if(cpu)ggml_backend_free(cpu);ggml_backend_free(backend);}
};
struct LayerGraph {
    LayerRun & r;int m,first,offset,read_start,s;bool full,placed=true;
    ggml_context * c=ctx(),*ic=ctx();ggml_backend_buffer_t ib=nullptr;
    ggml_backend_sched_t scheduler=nullptr;ggml_gallocr_t allocator=nullptr;ggml_cgraph * graph;
    ggml_tensor * x,*pos,*slots,*mask,*out;std::vector<std::map<std::string,ggml_tensor*>> stages;
    LayerGraph(LayerRun & run,int rows,int start,int position_offset,bool ffn,int window=0,int begin_layer=0,int end_layer=32):r(run),m(rows),first(start),offset(position_offset),read_start(window),s((start+rows-window+31)/32*32),full(ffn),stages(run.weights.size()){
        REQUIRE((m==1 || m==128) && first>=0 && read_start>=0 && read_start<=first && read_start+s<=r.capacity);
        x=ggml_new_tensor_2d(ic,GGML_TYPE_F32,4096,m);ggml_set_name(x,"tp_layer_input");pos=ggml_new_tensor_1d(ic,GGML_TYPE_I32,m);slots=ggml_new_tensor_1d(ic,GGML_TYPE_I64,m);mask=ggml_new_tensor_2d(ic,GGML_TYPE_F16,s,GGML_PAD(m,32));
        for(auto * t:{x,pos,slots,mask})ggml_set_input(t);ib=ggml_backend_alloc_ctx_tensors(ic,r.backend);REQUIRE(ib);set_input(0);
        graph=ggml_new_graph_custom(c,4096,false);ggml_tensor * current=x;
        for(size_t l=begin_layer;l<r.weights.size() && l<(size_t)end_layer;++l){auto & w=r.weights[l];auto keep=[&](const char * name,ggml_tensor * t){ggml_format_name(t,"%s-%zu",name,l);if(begin_layer>0 || end_layer<32 || r.weights.size()<=2 || !std::strcmp(name,"layer_out") || !std::strcmp(name,"attn_residual") || ((std::getenv("KUNLUN_TP_ACCURACY_TRACE")) && (!std::strcmp(name,"attn_norm") || !std::strcmp(name,"flash") || !std::strcmp(name,"wo") || !std::strcmp(name,"ffn_norm") || !std::strcmp(name,"down"))))ggml_set_output(t);stages[l][name]=t;return t;};
            auto * norm=keep("attn_norm",ggml_mul(c,ggml_rms_norm(c,current,1e-5f),w[AN]));
            auto * q=keep("q",ggml_mul_mat(c,w[WQ],norm)),*k=keep("k",ggml_mul_mat(c,w[WK],norm)),*v=keep("v",ggml_mul_mat(c,w[WV],norm));
            q=ggml_reshape_3d(c,q,128,32,m);k=ggml_reshape_3d(c,k,128,32,m);v=ggml_reshape_3d(c,v,128,32,m);
            q=keep("q_rope",ggml_rope_ext(c,q,pos,nullptr,128,0,4096,10000,1,0,1,32,1));k=keep("k_rope",ggml_rope_ext(c,k,pos,nullptr,128,0,4096,10000,1,0,1,32,1));
            ggml_build_forward_expand(graph,q);ggml_build_forward_expand(graph,v);ggml_build_forward_expand(graph,k);
            auto * kw=ggml_set_rows(c,r.k[l],ggml_view_2d(c,k,4096,m,k->nb[2],0),slots);auto * vw=ggml_set_rows(c,r.v[l],ggml_view_2d(c,v,4096,m,v->nb[2],0),slots);
            ggml_format_name(kw,"kv_k_write-%zu",l);ggml_format_name(vw,"kv_v_write-%zu",l);ggml_build_forward_expand(graph,kw);ggml_build_forward_expand(graph,vw);
            auto view=[&](ggml_tensor * cache){return ggml_view_4d(c,cache,128,32,s,1,256,8192,(size_t)8192*r.capacity,(size_t)8192*read_start);};
            q=keep("fa_q",ggml_permute(c,q,0,2,1,3));auto * kc=keep("fa_k",ggml_permute(c,view(r.k[l]),0,2,1,3));auto * vc=keep("fa_v",ggml_permute(c,view(r.v[l]),0,2,1,3));
            auto * fa=keep("flash",ggml_flash_attn_ext(c,q,kc,vc,mask,1.0f/std::sqrt(128.0f),0,0));ggml_prec_set_acc(fa,GGML_PREC_F32);
            auto * flat=keep("attn_flat",ggml_reshape_2d(c,fa,4096,m));auto * wo=keep("wo",ggml_mul_mat(c,w[WO],flat));auto * res=keep("attn_residual",ggml_add(c,wo,current));
            if(full){auto * fn=keep("ffn_norm",ggml_mul(c,ggml_rms_norm(c,res,1e-5f),w[FN]));auto * gate=keep("gate",ggml_mul_mat(c,w[WG],fn)),*up=keep("up",ggml_mul_mat(c,w[WU],fn));auto * glu=keep("glu",ggml_swiglu_split(c,gate,up));auto * down=keep("down",ggml_mul_mat(c,w[WD],glu));current=keep("layer_out",ggml_add(c,down,res));}
            else current=res;
        }
        out=current;ggml_build_forward_expand(graph,out);
        if(r.meta){
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto * t=ggml_graph_node(graph,i);if(!ggml_backend_supports_op(r.backend,t)){placed=false;std::printf("PLACEMENT_DECLINED node=%s op=%s shape=%lld,%lld,%lld,%lld\n",t->name,ggml_op_name(t->op),(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3]);}}
            if(!placed)return;
            ggml_backend_t bs[2]={r.backend,r.cpu};scheduler=ggml_backend_sched_new(bs,nullptr,2,4096,false,true);REQUIRE(scheduler && ggml_backend_sched_alloc_graph(scheduler,graph));
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto * t=ggml_graph_node(graph,i);REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler,t)==r.backend);}
            REQUIRE(ggml_backend_sched_get_n_splits(scheduler)==1);std::puts("PASS automatic Attention/Layer Meta placement, CPU splits=0");
            for(size_t l=0;l<stages.size();++l)if(!stages[l].empty())for(auto name:{"q","q_rope","fa_q","fa_k","flash","attn_flat","wo"}){auto * logical=stages[l][name];ggml_backend_meta_split_state ss{};REQUIRE(ggml_backend_meta_tensor_split_state(logical,false,&ss));
                for(int d=0;d<2;++d){auto * t=ggml_backend_meta_tensor_child(logical,d);REQUIRE(t);std::printf("SHARD %s PD%d axis=%d ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu offset=%zu root=%s\n",logical->name,d,ss.axis,(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],t->nb[0],t->nb[1],t->nb[2],t->nb[3],t->view_offs,t->view_src?t->view_src->name:"self");}}
        }else{allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(r.backend));REQUIRE(ggml_gallocr_alloc_graph(allocator,graph));}
    }
    void set_input(int variant){std::vector<float> input(4096*m);std::vector<int32_t> positions(m);std::vector<int64_t> indices(m);std::vector<ggml_fp16_t> masks(ggml_nelements(mask));
        for(size_t i=0;i<input.size();++i)input[i]=std::sin((i+(size_t)4096*first)*0.017+variant*0.1)*0.3f+std::cos(i*0.003+variant*0.2)*0.1f;
        for(int i=0;i<m;++i){positions[i]=offset+first+i;indices[i]=first+i;}
        for(int row=0;row<mask->ne[1];++row)for(int j=0;j<s;++j)masks[row*s+j]=ggml_fp32_to_fp16(row<m && read_start+j<=first+row?0:-INFINITY);
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(pos,positions.data(),0,m*4);ggml_backend_tensor_set(slots,indices.data(),0,m*8);ggml_backend_tensor_set(mask,masks.data(),0,masks.size()*2);
    }
    double compute(){REQUIRE(placed);auto begin=std::chrono::steady_clock::now();REQUIRE((scheduler?ggml_backend_sched_graph_compute(scheduler,graph):ggml_backend_graph_compute(r.backend,graph))==GGML_STATUS_SUCCESS);return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
    ~LayerGraph(){if(scheduler)ggml_backend_sched_free(scheduler);if(allocator)ggml_gallocr_free(allocator);ggml_free(c);ggml_backend_buffer_free(ib);ggml_free(ic);}
};
static std::vector<float> result(LayerRun & r,ggml_tensor * t){ggml_backend_meta_split_state ss{};if(r.meta && ggml_backend_meta_tensor_split_state(t,false,&ss) && ss.axis==GGML_BACKEND_SPLIT_AXIS_PARTIAL)return read(ggml_backend_meta_tensor_child(t,0));return read(t);}
static void compare_layers(LayerGraph & a,LayerGraph & b){
    if(std::getenv("KUNLUN_TP_ACCURACY_TRACE"))for(size_t l=0;l<a.stages.size();++l){auto * x=l?a.stages[l-1].at("layer_out"):a.x;auto * y=l?b.stages[l-1].at("layer_out"):b.x;compare(("layer"+std::to_string(l)+"/input").c_str(),result(a.r,x),result(b.r,y),std::numeric_limits<double>::infinity());}
    for(size_t l=0;l<a.stages.size();++l)for(auto & entry:a.stages[l]){if(entry.first.rfind("fa_",0)==0 || entry.first=="attn_flat")continue;auto * t=entry.second;auto * other=b.stages[l].at(entry.first);
        if(a.r.weights.size()>2 && !(t->flags&GGML_TENSOR_FLAG_OUTPUT))continue;
        double tolerance=l==0 && (entry.first=="q" || entry.first=="k" || entry.first=="v" || entry.first=="q_rope" || entry.first=="k_rope")?0.002:0.02;
        if(a.r.weights.size()>2)tolerance=std::numeric_limits<double>::infinity();
        compare(("layer"+std::to_string(l)+"/"+entry.first).c_str(),result(a.r,t),result(b.r,other),tolerance);
        if(entry.first=="wo" || entry.first=="down")REQUIRE(read(ggml_backend_meta_tensor_child(t,0))==read(ggml_backend_meta_tensor_child(t,1)));
    }
    for(size_t l=0;l<a.r.k.size();++l)for(int kind=0;kind<2;++kind){auto * x=kind?a.r.v[l]:a.r.k[l];auto * y=kind?b.r.v[l]:b.r.k[l];std::vector<ggml_fp16_t> ax(ggml_nelements(x)),by(ax.size());ggml_backend_tensor_get(x,ax.data(),0,ax.size()*2);ggml_backend_tensor_get(y,by.data(),0,by.size()*2);
        std::vector<float> af(ax.size()),bf(by.size());for(size_t i=0;i<ax.size();++i){af[i]=ggml_fp16_to_fp32(ax[i]);bf[i]=ggml_fp16_to_fp32(by[i]);}compare((kind?"KV/V":"KV/K"),af,bf,a.r.weights.size()>2?std::numeric_limits<double>::infinity():0.02);
        if(l==0){REQUIRE(ax==by);
            for(int d=0;d<2;++d){auto * child=ggml_backend_meta_tensor_child(x,d);std::vector<ggml_fp16_t> values(ggml_nelements(child));ggml_backend_tensor_get(child,values.data(),0,values.size()*2);
                for(int row=0;row<a.r.capacity;++row)REQUIRE(std::memcmp(values.data()+(size_t)row*2048,by.data()+(size_t)row*4096+d*2048,4096)==0);
            }
        }
    }
    a.r.check_guards();b.r.check_guards();
}
static void layer_single_parts(LayerGraph & g) {
    std::map<std::string,std::vector<ggml_tensor*>> groups;
    for(int i=0;i<ggml_graph_n_nodes(g.graph);++i){auto * t=ggml_graph_node(g.graph,i);std::string key;
        if(t->op==GGML_OP_MUL_MAT){std::string w=t->src[0]->name;if(w.find("attn_output")!=std::string::npos)key="WO";else if(w.find("attn_")!=std::string::npos)key="QKV";else if(w.find("ffn_down")!=std::string::npos)key="Down";else key="GateUp";}
        else if(t->op==GGML_OP_FLASH_ATTN_EXT)key="Flash";
        else if(t->op==GGML_OP_ROPE)key="RoPE";
        else if(t->op==GGML_OP_SET_ROWS)key="KV";
        else if(t->op==GGML_OP_GLU)key="SwiGLU";
        else if(t->op==GGML_OP_ADD)key="Residual";
        else if(t->op==GGML_OP_RMS_NORM || t->op==GGML_OP_MUL)key="Norm";
        if(!key.empty())groups[key].push_back(t);
    }
    g.compute();
    using Act=kunlun::CommonQ4ActivationStatistics (*)(ggml_backend_t);auto query=(Act)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_activation_statistics");REQUIRE(query);
    for(int rep=0;rep<2;++rep)for(auto & entry:groups){ggml_cgraph part=*g.graph;part.n_nodes=entry.second.size();part.nodes=entry.second.data();part.uid=0;auto q0=query(g.r.backend);auto begin=std::chrono::steady_clock::now();REQUIRE(ggml_backend_graph_compute(g.r.backend,&part)==GGML_STATUS_SUCCESS);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();auto q1=query(g.r.backend);
        std::printf("SINGLE_LAYER_PART M=%d rep=%d part=%s ms=%.6f quant_ms=%.6f\n",g.m,rep,entry.first.c_str(),ms,(q1.quantization_ns-q0.quantization_ns)/1e6);
    }
}
#include "attention-validation.inc"
#include "negative-validation.inc"
int main(int argc,char ** argv){
    std::setvbuf(stdout,nullptr,_IOLBF,0);REQUIRE(argc>=3);int m=std::atoi(argv[2]),layers=argc>3?std::atoi(argv[3]):1;REQUIRE((m==1 || m==128) && layers>0 && layers<=32);
    bool placement=argc>4 && !std::strcmp(argv[4],"placement"),attention=argc>4 && !std::strcmp(argv[4],"attention");
    setenv("GGML_KUNLUN_TENSOR_PARALLEL","1",1);setenv("GGML_KUNLUN_TP_FFN","1",1);setenv("GGML_KUNLUN_TP_ATTENTION","1",1);
    LayerData data(argv[1],layers);auto initial_hbm=hbm();
    { LayerRun single(data,false);
    if(argc>4 && !std::strcmp(argv[4],"single-parts")){LayerGraph b(single,m,0,37,true);layer_single_parts(b);std::puts("PASS Single Layer parts");return 0;}
    LayerRun tp(data,true);
    if(argc>4 && !std::strcmp(argv[4],"seed-control")) {
        REQUIRE(layers==32 && m==128);
        LayerGraph a(tp,m,0,37,true,0,0,1),b(single,m,0,37,true);
        b.compute();auto baseline=result(single,b.out);a.compute();auto seed=result(tp,a.stages[0].at("wo"));
        auto original=result(single,b.stages[0].at("wo"));compare("WO_ROUNDOFF_SEED",seed,original,1e-6);
        int cut=-1;for(int i=0;i<ggml_graph_n_nodes(b.graph);++i)if(ggml_graph_node(b.graph,i)==b.stages[0].at("wo"))cut=i+1;REQUIRE(cut>0);
        ggml_cgraph prefix=*b.graph;prefix.uid=0;prefix.n_nodes=cut;REQUIRE(ggml_backend_graph_compute(single.backend,&prefix)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_set(b.stages[0].at("wo"),seed.data(),0,seed.size()*4);
        ggml_cgraph suffix=*b.graph;suffix.uid=0;suffix.nodes+=cut;suffix.n_nodes-=cut;REQUIRE(ggml_backend_graph_compute(single.backend,&suffix)==GGML_STATUS_SUCCESS);
        compare("SINGLE_32_WITH_ONLY_TP_FIRST_WO",result(single,b.out),baseline,std::numeric_limits<double>::infinity());
        single.check_guards();tp.check_guards();std::puts("DEEP_FINITE_COMPLETE WO_ROUNDOFF_SEED_CONTROL");return 0;
    }
    if(argc>4 && !std::strcmp(argv[4],"local-control")) {
        std::vector<float> common;
        for(int layer=0;layer<layers;++layer) {
            LayerGraph a(tp,m,0,37,true,0,layer,layer+1),b(single,m,0,37,true,0,layer,layer+1);
            if(layer){ggml_backend_tensor_set(a.x,common.data(),0,common.size()*4);ggml_backend_tensor_set(b.x,common.data(),0,common.size()*4);}
            b.compute();a.compute();
            for(const char *name:{"attn_norm","flash","wo","attn_residual","ffn_norm","down","layer_out"}) {
                auto x=result(tp,a.stages[layer].at(name)),y=result(single,b.stages[layer].at(name));
                compare(("LOCAL_SAME_INPUT layer"+std::to_string(layer)+"/"+name).c_str(),x,y,std::numeric_limits<double>::infinity());
            }
            using Query=kunlun::CommonQ4Quantization (*)(ggml_backend_t);
            auto query=(Query)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_quantization");REQUIRE(query);
            auto full=snapshot(0,query(single.backend));QuantCopy locals[2]={snapshot(0,query(ggml_backend_meta_simple_backend(tp.backend,0))),snapshot(1,query(ggml_backend_meta_simple_backend(tp.backend,1)))};size_t differing=0,scale_rows=0;
            for(int row=0;row<m;++row){bool diff=false;for(int d=0;d<2;++d){const auto &q=locals[d];
                REQUIRE(q.kp==5632 && full.kp==11008 && q.components==1 && full.components==1);
                diff=diff || std::memcmp(q.params.data()+row*64,full.params.data()+row*64,64)!=0;
                for(int j=0;j<5504;++j)differing+=q.q[row*q.kp+j]!=full.q[row*full.kp+d*5504+j];
            }scale_rows+=diff;}
            std::printf("QUANT_BOUNDARY layer=%d different_bytes=%zu/%d different_scale_rows=%zu/%d\n",layer,differing,11008*m,scale_rows,m);
            common=result(single,b.out);tp.check_guards();single.check_guards();
        }
        std::puts("DEEP_FINITE_COMPLETE LOCAL_SAME_INPUT_CONTROL");return 0;
    }
    if(m==1 && !placement)packed_attention(tp,data);
    {LayerGraph a(tp,m,0,37,!attention),b(single,m,0,37,!attention);if(placement){std::printf("PLACEMENT_RESULT %d\n",(int)a.placed);return 0;}
        std::vector<float> first;for(int rep=0;rep<(layers>2?1:3);++rep){auto before=kunlun::statistics();auto cpu=std::clock();double sm=b.compute();double scpu=1000.0*(std::clock()-cpu)/CLOCKS_PER_SEC;using Profile=ggml_kunlun_tp_profile (*)(ggml_backend_t);auto query=(Profile)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_profile");REQUIRE(query);auto * child=ggml_backend_meta_simple_backend(tp.backend,0);using QS=kunlun::CommonQ4TPStatistics (*)(ggml_backend_t);auto quant_query=(QS)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_quant_statistics");REQUIRE(quant_query);auto q0=quant_query(child);auto p0=query(child);auto wire0=transport();cpu=std::clock();double tm=a.compute();double tcpu=1000.0*(std::clock()-cpu)/CLOCKS_PER_SEC;auto p1=query(child);auto q1=quant_query(child);std::printf("LAYER_QUANT M=%d rep=%d global_prepare_ms=%.6f max_ms=%.6f max_rounds=%llu max_p2p=%llu max_bytes=%llu\n",m,rep,(q1.prepare_ns-q0.prepare_ns)/1e6,(q1.max_ns-q0.max_ns)/1e6,(unsigned long long)(q1.max_collectives-q0.max_collectives),(unsigned long long)(q1.peer_copies-q0.peer_copies),(unsigned long long)(q1.peer_bytes-q0.peer_bytes));auto after=kunlun::statistics();auto wire1=transport();
            if(rep>0){unsigned long long calls=layers*(m==1?(attention?6:12):(attention?4:8)),bytes=layers*(m==1?(attention?33792ull:68608ull):(attention?4227072ull:8519680ull));
                const char *global_max=std::getenv("GGML_KUNLUN_TP_FA_GLOBAL_MAX");
                if(global_max && !std::strcmp(global_max,"1")){const char *component=std::getenv("KUNLUN_TP_FA_MAX_COMPONENT");calls+=2*layers;bytes+=(component?512ull:1024ull)*layers;}
                REQUIRE(wire1[0]-wire0[0]==calls && wire1[1]-wire0[1]==bytes && wire1[2]==wire0[2] && wire1[3]==wire0[3]);
                std::printf("LAYER_TRANSPORT M=%d rep=%d SSE_calls=%llu SSE_bytes=%llu Host_calls=0 Host_bytes=0\n",m,rep,calls,bytes);
            }

            std::printf("LAYER_TIME M=%d layers=%d attention=%d rep=%d single_ms=%.6f tp_ms=%.6f single_cpu_ms=%.6f tp_cpu_ms=%.6f p2p=%zu bytes=%zu hbm0=%zu hbm1=%zu\n",m,layers,(int)attention,rep,sm,tm,scpu,tcpu,after.peer_copies-before.peer_copies,after.peer_bytes-before.peer_bytes,kunlun::allocated_bytes(0),kunlun::allocated_bytes(1));
            for(int phase=0;phase<TP_PHASE_COUNT;++phase)std::printf("LAYER_PHASE M=%d rep=%d phase=%d ms=%.6f calls=%llu\n",m,rep,phase,(p1.ns[phase]-p0.ns[phase])/1e6,(unsigned long long)(p1.calls[phase]-p0.calls[phase]));
            std::printf("LAYER_WAIT M=%d rep=%d q4_finish_ms=%.6f attention_wait_pd0_ms=%.6f attention_wait_pd1_ms=%.6f\n",m,rep,(p1.finish_wait_ns-p0.finish_wait_ns)/1e6,(p1.attention_wait_ns[0]-p0.attention_wait_ns[0])/1e6,(p1.attention_wait_ns[1]-p0.attention_wait_ns[1])/1e6);
            if(layers==1 && rep==1) {
                using Inspect=kunlun::FlashInspection (*)(ggml_backend_t);auto inspect=(Inspect)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_flash_inspection");REQUIRE(inspect);
                auto full=inspect(single.backend);auto fp=device_read<float>(0,full.peaks,128);auto fr=device_read<float>(0,full.ranges,32*8);
                for(int d=0;d<2;++d){auto local=inspect(ggml_backend_meta_simple_backend(tp.backend,d));auto lp=device_read<float>(d,local.peaks,128);auto lr=device_read<float>(d,local.ranges,16*8);bool ranges=true;
                    for(int h=0;h<16;++h)ranges=ranges && lr[h*8]==fr[(d*16+h)*8];
                    std::printf("FA_MAX M=%d PD=%d SingleK=%.9g LocalK=%.9g SingleV=%.9g LocalV=%.9g Qrange_equal=%d\n",m,d,*std::max_element(fp.begin(),fp.begin()+4),*std::max_element(lp.begin(),lp.begin()+4),*std::max_element(fp.begin()+64,fp.begin()+68),*std::max_element(lp.begin()+64,lp.begin()+68),(int)ranges);
                    REQUIRE(ranges);
                }
            }
            compare_layers(a,b);auto out=result(tp,a.out);if(rep)REQUIRE(out==first);else first=out;
        }
        if(attention){wo_quantization(a,data,b);rejected_attention(a);}
    }
    for(int start:{m,m+1,m+2}){LayerGraph a(tp,1,start,37,!attention),b(single,1,start,37,!attention);double step_single=b.compute(),step_tp=a.compute();std::printf("CACHE_STEP M=%d first=%d read_start=%d S=%d layers=%d single_ms=%.6f tp_ms=%.6f\n",a.m,a.first,a.read_start,a.s,layers,step_single,step_tp);compare_layers(a,b);}
    std::puts("PASS Prefill -> 3 continuous Decode / graph rebuild / position offset / cache guards");
    bool extended=argc>4 && !std::strcmp(argv[4],"extended");
    if(extended && m==128) {
        for(int start:{128,256,384}){LayerGraph a(tp,128,start,37,true),b(single,128,start,37,true);double step_single=b.compute(),step_tp=a.compute();std::printf("CACHE_STEP M=%d first=%d read_start=%d S=%d layers=%d single_ms=%.6f tp_ms=%.6f\n",a.m,a.first,a.read_start,a.s,layers,step_single,step_tp);compare_layers(a,b);}
        auto before=kunlun::kv_max_statistics(1);
        for(int start:{512,513,514}){LayerGraph a(tp,1,start,37,true),b(single,1,start,37,true);double step_single=b.compute(),step_tp=a.compute();std::printf("CACHE_STEP M=%d first=%d read_start=%d S=%d layers=%d single_ms=%.6f tp_ms=%.6f\n",a.m,a.first,a.read_start,a.s,layers,step_single,step_tp);compare_layers(a,b);}
        {LayerGraph a(tp,1,515,37,true,32),b(single,1,515,37,true,32);double step_single=b.compute(),step_tp=a.compute();std::printf("CACHE_STEP M=%d first=%d read_start=%d S=%d layers=%d single_ms=%.6f tp_ms=%.6f\n",a.m,a.first,a.read_start,a.s,layers,step_single,step_tp);compare_layers(a,b);}
        auto after=kunlun::kv_max_statistics(1);REQUIRE(after.hits>before.hits && after.fallback_unowned==before.fallback_unowned && after.verify_mismatches==0);
        std::printf("PASS KV512 plus Decode / nonzero physical View offset / PD1 max-cache hits=%zu\n",after.hits-before.hits);
    }

    if(layers>2){std::puts("DEEP_FINITE_COMPLETE (stage errors above; no numerical acceptance claim)");return 0;}
    tp.reset_cache();single.reset_cache();{LayerGraph a(tp,m,0,13,!attention),b(single,m,0,13,!attention);a.set_input(1);b.set_input(1);double step_single=b.compute(),step_tp=a.compute();std::printf("CACHE_STEP M=%d first=%d read_start=%d S=%d layers=%d single_ms=%.6f tp_ms=%.6f\n",a.m,a.first,a.read_start,a.s,layers,step_single,step_tp);compare_layers(a,b);}
    std::puts("PASS cache reallocation / new generation / new input");
    }
    REQUIRE(hbm()==initial_hbm);
    if(argc>4 && !std::strcmp(argv[4],"extended")){
        LayerRun single(data,false),tp(data,true);LayerGraph a(tp,1,0,91,true),b(single,1,0,91,true);a.set_input(2);b.set_input(2);double step_single=b.compute(),step_tp=a.compute();std::printf("CACHE_STEP M=%d first=%d read_start=%d S=%d layers=%d single_ms=%.6f tp_ms=%.6f\n",a.m,a.first,a.read_start,a.s,layers,step_single,step_tp);compare_layers(a,b);
    }
    REQUIRE(hbm()==initial_hbm);
    std::puts("PASS graph/context destruction returns tracked HBM to baseline");
    std::puts("PASS Phase3 Attention and Layer");
}
