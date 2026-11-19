#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"
#include "ggml-kunlun.h"
#include "llama-model.h"
#include <kunlun/k200.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <memory>
#include <limits>
#include <ctime>
#include <array>
#include "tp-profile.h"

#define REQUIRE(x) do{if(!(x)){std::fprintf(stderr,"FAIL %d %s\n",__LINE__,#x);std::exit(1);}}while(0)
static ggml_context * ctx(){return ggml_init({16*1024*1024,nullptr,true});}
static std::vector<float> read(ggml_tensor * t){std::vector<float> v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
static void compare(const char * name,const std::vector<float> & a,const std::vector<float> & b,double tolerance=1e-3){
    REQUIRE(a.size()==b.size());double diff=0,ref=0,maxdiff=0,maxref=0;size_t unequal=0;
    for(size_t i=0;i<a.size();++i){REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));double d=(double)a[i]-b[i];diff+=d*d;ref+=(double)b[i]*b[i];maxdiff=std::max(maxdiff,std::abs(d));maxref=std::max(maxref,(double)std::abs(b[i]));unequal+=std::memcmp(&a[i],&b[i],4)!=0;}
    double relative=std::sqrt(diff/std::max(ref,1e-30));std::printf("ACCURACY %s rel_l2=%.9g max_abs=%.9g max_relative=%.9g unequal=%zu/%zu\n",name,relative,maxdiff,maxdiff/std::max(maxref,1e-30),unequal,a.size());REQUIRE(relative<=tolerance);
}
struct ModelData {
    ggml_context * info=nullptr;gguf_context * gguf=nullptr;
    std::vector<unsigned char> raw[3];int64_t k[3]={4096,4096,11008},n[3]={11008,11008,4096};
    const char * names[3]={"blk.0.ffn_gate.weight","blk.0.ffn_up.weight","blk.0.ffn_down.weight"};
    explicit ModelData(const char * path){gguf=gguf_init_from_file(path,{true,&info});REQUIRE(gguf && info);std::ifstream file(path,std::ios::binary);REQUIRE(file.good());
        for(int i=0;i<3;++i){int64_t id=gguf_find_tensor(gguf,names[i]);REQUIRE(id>=0);auto * t=ggml_get_tensor(info,names[i]);REQUIRE(t && t->type==GGML_TYPE_Q4_0 && t->ne[0]==k[i] && t->ne[1]==n[i]);raw[i].resize(ggml_nbytes(t));file.seekg(gguf_get_data_offset(gguf)+gguf_get_tensor_offset(gguf,id));file.read((char *)raw[i].data(),raw[i].size());REQUIRE((size_t)file.gcount()==raw[i].size());}}
    ~ModelData(){ggml_free(info);gguf_free(gguf);}
};
struct Run {
    ggml_backend_t backend,cpu=nullptr;ggml_context * weights=ctx();ggml_backend_buffer_t storage=nullptr;
    ggml_tensor * w[3],*x;std::unique_ptr<llama_model> model;
    llama_meta_device_get_split_state_userdata policy{};bool meta;int rows;
    Run(ModelData & data,int m,bool split):meta(split),rows(m){
        for(int i=0;i<3;++i){w[i]=ggml_new_tensor_2d(weights,GGML_TYPE_Q4_0,data.k[i],data.n[i]);ggml_set_name(w[i],data.names[i]);}
        x=ggml_new_tensor_2d(weights,GGML_TYPE_F32,4096,m);ggml_set_name(x,"ffn_input");ggml_set_input(x);
        if(split){
            model.reset(llama_model_create(LLM_ARCH_LLAMA,llama_model_default_params()));REQUIRE(model);
            model->hparams.n_embd=4096;model->hparams.n_layer_all=32;model->hparams.n_ff_arr.fill(11008);model->hparams.n_head_arr.fill(32);model->hparams.n_head_kv_arr.fill(32);
            model->hparams.n_embd_head_k_full=128;model->hparams.n_embd_head_v_full=128;
            for(int i=0;i<3;++i)model->tensors_by_name.emplace_back(data.names[i],w[i]);
            policy={2,model.get()};
            auto reg=ggml_backend_kunlun_reg();ggml_backend_dev_t devices[2]={ggml_backend_reg_dev_get(reg,0),ggml_backend_reg_dev_get(reg,1)};
            backend=ggml_backend_dev_init(ggml_backend_meta_device(devices,2,llama_meta_device_get_split_state,&policy),nullptr);
            cpu=ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),nullptr);
        }else backend=ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_kunlun_reg(),0),nullptr);
        REQUIRE(backend);storage=ggml_backend_alloc_ctx_tensors(weights,backend);REQUIRE(storage);
        for(int i=0;i<3;++i){ggml_backend_tensor_set(w[i],data.raw[i].data(),0,data.raw[i].size());std::vector<unsigned char> restored(data.raw[i].size());ggml_backend_tensor_get(w[i],restored.data(),0,restored.size());REQUIRE(restored==data.raw[i]);}
        std::printf("PASS weights %s raw block roundtrip\n",split?"Meta":"Single");
    }
    ~Run(){ggml_backend_buffer_free(storage);ggml_free(weights);if(cpu)ggml_backend_free(cpu);ggml_backend_free(backend);}
};
struct Graph {
    Run & run;ggml_context * c=ctx();ggml_tensor * gate,*up,*glu=nullptr,*down=nullptr,*out=nullptr;
    ggml_cgraph * graph;ggml_backend_sched_t scheduler=nullptr;ggml_gallocr_t alloc=nullptr;
    Graph(Run & r,bool full):run(r){gate=ggml_mul_mat(c,r.w[0],r.x);up=ggml_mul_mat(c,r.w[1],r.x);ggml_set_name(gate,"ffn_gate");ggml_set_name(up,"ffn_up");ggml_set_output(gate);ggml_set_output(up);
        graph=ggml_new_graph(c);
        if(full){glu=ggml_swiglu_split(c,gate,up);ggml_set_name(glu,"ffn_glu");ggml_set_output(glu);down=ggml_mul_mat(c,r.w[2],glu);ggml_set_name(down,"ffn_down");ggml_set_output(down);out=ggml_add(c,down,r.x);ggml_set_name(out,"ffn_residual");ggml_set_output(out);ggml_build_forward_expand(graph,out);}
        else{ggml_build_forward_expand(graph,gate);ggml_build_forward_expand(graph,up);}
        if(r.meta){
            for(auto * t:{gate,up,glu,down,out})if(t)REQUIRE(ggml_backend_supports_op(r.backend,t));
            ggml_backend_t bs[2]={r.backend,r.cpu};scheduler=ggml_backend_sched_new(bs,nullptr,2,2048,false,true);REQUIRE(scheduler && ggml_backend_sched_alloc_graph(scheduler,graph));
            for(auto * t:{gate,up,glu,down,out})if(t)REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler,t)==r.backend);
            REQUIRE(ggml_backend_sched_get_n_splits(scheduler)==1);std::puts("PASS automatic Meta placement (no CPU graph)");
        }else{alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(r.backend));REQUIRE(ggml_gallocr_alloc_graph(alloc,graph));}
    }
    ~Graph(){if(scheduler)ggml_backend_sched_free(scheduler);if(alloc)ggml_gallocr_free(alloc);ggml_free(c);}
    double compute(){auto begin=std::chrono::steady_clock::now();auto status=scheduler?ggml_backend_sched_graph_compute(scheduler,graph):ggml_backend_graph_compute(run.backend,graph);REQUIRE(status==GGML_STATUS_SUCCESS);return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
};
#include "ffn-validation.inc"
static std::array<size_t,2> hbm(){return {kunlun::allocated_bytes(0),kunlun::allocated_bytes(1)};}
static std::array<unsigned long long,4> transport(){
    std::ifstream f("/proc/xpu/dev0/p2p_stats");REQUIRE(f.good());
    std::array<unsigned long long,4> v{};std::string line;
    while(std::getline(f,line))for(int i=0;i<4;++i){const char * names[]={"sse_calls=","sse_bytes=","host_calls=","host_bytes="};if(line.rfind(names[i],0)==0)v[i]=std::stoull(line.substr(std::strlen(names[i])));}
    return v;
}
static void single_parts(Run & run,Graph & g) {
    REQUIRE(ggml_graph_n_nodes(g.graph)==5);
    using Act=kunlun::CommonQ4ActivationStatistics (*)(ggml_backend_t);
    auto query=(Act)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_activation_statistics");REQUIRE(query);
    g.compute();
    for(int rep=0;rep<2;++rep)for(int part=0;part<4;++part){
        int start=part==0?0:part+1,stop=part==0?2:part+2;ggml_cgraph one=*g.graph;one.n_nodes=stop-start;one.nodes=g.graph->nodes+start;one.uid=0;
        auto before=query(run.backend);auto begin=std::chrono::steady_clock::now();REQUIRE(ggml_backend_graph_compute(run.backend,&one)==GGML_STATUS_SUCCESS);
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();auto after=query(run.backend);
        std::printf("SINGLE_PART M=%d rep=%d part=%d wall_ms=%.6f quant_ms=%.6f\n",run.rows,rep,part,ms,(after.quantization_ns-before.quantization_ns)/1e6);
    }
}
#ifndef KUNLUN_FFN_TEST_NO_MAIN
int main(int argc,char ** argv){
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    REQUIRE(argc>=3);int m=std::atoi(argv[2]);REQUIRE(m==1 || m==128);bool full=argc>3 && !std::strcmp(argv[3],"full");
    setenv("GGML_KUNLUN_TENSOR_PARALLEL","1",1);setenv("GGML_KUNLUN_TP_FFN","1",1);
    ModelData data(argv[1]);auto initial=hbm();Run single(data,m,false);auto sw=hbm();
    bool parts=argc>4 && !std::strcmp(argv[4],"single-parts");
    if(parts){std::vector<float> input(4096*m);for(size_t i=0;i<input.size();++i)input[i]=std::sin(i*0.017)*0.3f+std::cos(i*0.003)*0.1f;ggml_backend_tensor_set(single.x,input.data(),0,input.size()*4);Graph g(single,true);single_parts(single,g);std::puts("PASS Single parts");return 0;}
    Run tp(data,m,true);auto tw=hbm();
    std::printf("MEMORY weights+input single0=%zu single1=%zu tp0=%zu tp1=%zu\n",sw[0]-initial[0],sw[1]-initial[1],tw[0]-sw[0],tw[1]-sw[1]);
    if(m==1)verify_packed(tp);
    std::vector<float> x(4096*m);for(size_t i=0;i<x.size();++i)x[i]=std::sin(i*0.017)*0.3f+std::cos(i*0.003)*0.1f;
    for(auto * r:{&single,&tp})ggml_backend_tensor_set(r->x,x.data(),0,x.size()*4);
    Graph gs(single,full);auto sg=hbm();Graph gt(tp,full);auto tg=hbm();
    using Profile=ggml_kunlun_tp_profile (*)(ggml_backend_t);
    using QuantStats=kunlun::CommonQ4TPStatistics (*)(ggml_backend_t);
    auto profile=(Profile)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_profile");
    auto quant_stats=(QuantStats)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_quant_statistics");REQUIRE(profile && quant_stats);
    auto * simple0=ggml_backend_meta_simple_backend(tp.backend,0);
    double tolerance=std::getenv("GGML_KUNLUN_TP_QUANT") && !std::strcmp(std::getenv("GGML_KUNLUN_TP_QUANT"),"local")?0.03:0.001;
    std::vector<float> first;
    for(int rep=0;rep<3;++rep){auto before=kunlun::statistics();auto cpu=std::clock();double a=gs.compute();double scpu=1000.0*(std::clock()-cpu)/CLOCKS_PER_SEC;auto h_before=hbm();auto p0=profile(simple0);auto q0=quant_stats(simple0);cpu=std::clock();double b=gt.compute();double tcpu=1000.0*(std::clock()-cpu)/CLOCKS_PER_SEC;auto after=kunlun::statistics();auto p1=profile(simple0);auto q1=quant_stats(simple0);auto h_after=hbm();
        std::printf("TIME M=%d full=%d rep=%d single_ms=%.6f tp_ms=%.6f single_cpu_ms=%.6f tp_cpu_ms=%.6f p2p=%zu bytes=%zu hbm0=%zu hbm1=%zu\n",m,(int)full,rep,a,b,scpu,tcpu,after.peer_copies-before.peer_copies,after.peer_bytes-before.peer_bytes,h_after[0],h_after[1]);
        for(int phase=0;phase<TP_PHASE_COUNT;++phase)std::printf("PHASE M=%d rep=%d phase=%d wall_ms=%.6f calls=%llu\n",m,rep,phase,(p1.ns[phase]-p0.ns[phase])/1e6,(unsigned long long)(p1.calls[phase]-p0.calls[phase]));
        std::printf("CONTROL M=%d rep=%d finish_wait_ms=%.6f max_ms=%.6f quant_prepare_ms=%.6f quant_wait_ms=%.6f max_calls=%llu max_p2p=%llu max_bytes=%llu pair=%llu\n",m,rep,(p1.finish_wait_ns-p0.finish_wait_ns)/1e6,(q1.max_ns-q0.max_ns)/1e6,(q1.prepare_ns-q0.prepare_ns)/1e6,(q1.wait_ns-q0.wait_ns)/1e6,(unsigned long long)(q1.max_collectives-q0.max_collectives),(unsigned long long)(q1.peer_copies-q0.peer_copies),(unsigned long long)(q1.peer_bytes-q0.peer_bytes),(unsigned long long)(p1.pair_calls-p0.pair_calls));
        if(!rep)std::printf("MEMORY estimated_TP_owned0=%zu estimated_TP_owned1=%zu (weights+input+graph+compute-delta; excludes shared L3)\n",(tw[0]-sw[0])+(tg[0]-sg[0])+(h_after[0]-h_before[0]),(tw[1]-sw[1])+(tg[1]-sg[1])+(h_after[1]-h_before[1]));
        compare("Gate",read(gt.gate),read(gs.gate));compare("Up",read(gt.up),read(gs.up));if(full){compare("SwiGLU",read(gt.glu),read(gs.glu));for(int d=0;d<2;++d)compare("Down",read(ggml_backend_meta_tensor_child(gt.down,d)),read(gs.down),tolerance);REQUIRE(read(ggml_backend_meta_tensor_child(gt.down,0))==read(ggml_backend_meta_tensor_child(gt.down,1)));auto result=read(gt.out);compare("FFN+residual",result,read(gs.out),tolerance);if(rep)REQUIRE(std::memcmp(result.data(),first.data(),result.size()*4)==0);else first=result;}}
    if(full && std::getenv("KUNLUN_TP_TRANSPORT_CHECK")) {
        auto a=transport();gt.compute();auto b=transport();
        bool local=std::getenv("GGML_KUNLUN_TP_QUANT") && !std::strcmp(std::getenv("GGML_KUNLUN_TP_QUANT"),"local");
        const unsigned long long calls=local?2:(m==1?6:4),bytes=local?2ull*4096*m*4:(m==1?34816:4292608);
        REQUIRE(b[0]-a[0]==calls && b[1]-a[1]==bytes && b[2]==a[2] && b[3]==a[3]);
        std::printf("PASS warm FFN driver SSE calls=%llu bytes=%llu Host calls=0 bytes=0\n",calls,bytes);
    }
    bool global=!std::getenv("GGML_KUNLUN_TP_QUANT") || !std::strcmp(std::getenv("GGML_KUNLUN_TP_QUANT"),"global");
    if(full && global){verify_graph_quant(tp,gt);partial_guards(tp,gt);quant_edges(m);if(m==1)cpu_reference(data,x,gt);}
    if(full){
        for(size_t i=0;i<x.size();++i)x[i]=std::cos(i*0.023+0.5)*0.2f;
        for(auto * r:{&single,&tp})ggml_backend_tensor_set(r->x,x.data(),0,x.size()*4);
        {Graph next_single(single,true),next_tp(tp,true);next_single.compute();next_tp.compute();compare("rebuilt graph + new input",read(next_tp.out),read(next_single.out),tolerance);REQUIRE(read(next_tp.out)!=first);if(global)verify_graph_quant(tp,next_tp);}
        ggml_tensor invalid=*gt.down;invalid.ne[0]--;REQUIRE(!ggml_backend_supports_op(tp.backend,&invalid));
        std::puts("PASS input generation / graph rebuild / allocation / invalid-shape probe");
    }
    std::puts("PASS FFN stage");
}

#endif
