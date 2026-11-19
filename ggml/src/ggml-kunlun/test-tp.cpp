#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-kunlun.h"
#include "../ggml-impl.h"
#include <kunlun/k200.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <map>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
static ggml_context * context() { return ggml_init({32*1024*1024,nullptr,true}); }
static ggml_backend_meta_split_state policy(const ggml_tensor * t,void *) {
    ggml_backend_meta_split_state s={GGML_BACKEND_SPLIT_AXIS_MIRRORED,{0},{1},1};
    int axis=-1;
    if (!std::strncmp(t->name,"axis",4)) axis=t->name[4]-'0';
    if (!std::strcmp(t->name,"weight") || !std::strcmp(t->name,"input")) axis=0;
    if (axis>=0 && axis<4) { s.axis=(ggml_backend_meta_split_axis)axis;s.ne[0]=t->ne[axis]/2;s.ne[1]=t->ne[axis]-s.ne[0]; }
    return s;
}
static void equal(const std::vector<float> & a,const std::vector<float> & b) {
    REQUIRE(a.size()==b.size());
    for (size_t i=0;i<a.size();++i) if (std::memcmp(&a[i],&b[i],4)) {
        std::fprintf(stderr,"mismatch %zu got=%.9g expected=%.9g\n",i,a[i],b[i]);REQUIRE(false);
    }
}
static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;
}
static std::map<std::string,uint64_t> counters() {
    std::ifstream f("/proc/xpu/dev0/p2p_stats");REQUIRE(f.good());
    std::map<std::string,uint64_t> out;std::string line;
    while(std::getline(f,line)){auto p=line.find('=');if(p!=std::string::npos) {auto key=line.substr(0,p);if(key=="sse_calls" || key=="host_calls" || key=="host_bytes" || key=="fallback_3_calls" || key=="fallback_3_bytes")out[key]=std::stoull(line.substr(p+1));}}
    REQUIRE(out.count("sse_calls") && out.count("host_calls"));return out;
}
static void placement(ggml_backend_t meta) {
    auto * c=context();
    std::vector<ggml_tensor *> ts;
    for (int axis=-1;axis<4;++axis) {
        auto * t=ggml_new_tensor_4d(c,GGML_TYPE_F32,64,8,4,2);
        ggml_format_name(t,axis<0?"mirrored":"axis%d",axis);ts.push_back(t);
    }
    auto * buf=ggml_backend_alloc_ctx_tensors(c,meta);REQUIRE(buf);
    for (int which=0;which<5;++which) {
        auto * t=ts[which];std::vector<float> v(ggml_nelements(t));
        for (size_t i=0;i<v.size();++i) v[i]=(float)(i-0.0);
        ggml_backend_tensor_set(t,v.data(),0,v.size()*4);equal(get(t),v);
        for (int d=0;d<2;++d) {
            auto * child=ggml_backend_meta_tensor_child(t,d);REQUIRE(child);
            REQUIRE(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(child->buffer))==ggml_backend_dev_by_name(d?"KUNLUN1":"KUNLUN0"));
            auto actual=get(child);size_t p=0;
            for (int i3=0;i3<child->ne[3];++i3) for (int i2=0;i2<child->ne[2];++i2)
            for (int i1=0;i1<child->ne[1];++i1) for (int i0=0;i0<child->ne[0];++i0) {
                int coord[4]={i0,i1,i2,i3};int axis=which-1;
                if (axis>=0) coord[axis]+=d*t->ne[axis]/2;
                size_t index=coord[0]+t->ne[0]*(coord[1]+t->ne[1]*(coord[2]+t->ne[2]*coord[3]));
                REQUIRE(actual[p++]==v[index]);
            }
        }
    }
    ggml_backend_buffer_free(buf);ggml_free(c);std::puts("PASS placement mirrored axis0/1/2/3 physical ownership roundtrip");
}
static void reduction(ggml_backend_reg_t reg) {
    ggml_backend_t backends[2]={ggml_backend_dev_init(ggml_backend_dev_by_name("KUNLUN0"),nullptr),ggml_backend_dev_init(ggml_backend_dev_by_name("KUNLUN1"),nullptr)};
    auto init=(ggml_backend_comm_init_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_init");
    auto sum=(ggml_backend_comm_allreduce_tensor_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_allreduce_tensor");
    auto free_comm=(ggml_backend_comm_free_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_free");
    REQUIRE(init && sum && free_comm);REQUIRE(init(backends,1)==nullptr);
    void * comm=init(backends,2);REQUIRE(comm);
    for (int rows:{1,128}) {
        auto driver_before=counters();
        auto * c=context();ggml_tensor * t[2];ggml_backend_buffer_t buf[2];
        const int count=4096*rows;
        std::vector<float> ref(count),initial[2];
        for (int d=0;d<2;++d) {
            // A view leaves a 256-byte guard before and after the valid payload.
            auto * root=ggml_new_tensor_1d(c,GGML_TYPE_F32,count+128);
            buf[d]=ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backends[d]),(count+128)*4);
            REQUIRE(buf[d]);REQUIRE(ggml_backend_tensor_alloc(buf[d],root,ggml_backend_buffer_get_base(buf[d]))==GGML_STATUS_SUCCESS);
            t[d]=ggml_view_2d(c,root,4096,rows,4096*4,256);REQUIRE(ggml_backend_view_init(t[d])==GGML_STATUS_SUCCESS);
            std::vector<float> all(count+128,-12345);for(int i=0;i<count;++i) all[i+64]=(float)((i%127)-63)*(d?0.25f:0.5f)+(float)d;
            initial[d]=std::vector<float>(all.begin()+64,all.begin()+64+count);
            ggml_backend_tensor_set(root,all.data(),0,all.size()*4);
        }
        for (int i=0;i<count;++i) ref[i]=initial[0][i]+initial[1][i];
        for (int rep=0;rep<3;++rep) {
            for (int d=0;d<2;++d)ggml_backend_tensor_set(t[d],initial[d].data(),0,count*4);
            auto before=kunlun::statistics();REQUIRE(sum(comm,t));auto after=kunlun::statistics();
            REQUIRE(after.peer_copies-before.peer_copies==2);REQUIRE(after.peer_bytes-before.peer_bytes==(size_t)count*8);
            equal(get(t[0]),ref);equal(get(t[1]),ref);
            for(int d=0;d<2;++d) {
                auto all=get(t[d]->view_src);for(int i=0;i<64;++i){REQUIRE(all[i]==-12345);REQUIRE(all[count+64+i]==-12345);}
            }
            if(rep==0) {
                auto warm=counters();
                REQUIRE(warm["sse_calls"]-driver_before["sse_calls"]==2);
                // XRE performs one 16-byte capability probe per direction on first use.
                uint64_t host=warm["host_calls"]-driver_before["host_calls"];
                REQUIRE(host==0 || (host==2 && warm["host_bytes"]-driver_before["host_bytes"]==32 &&
                    warm["fallback_3_calls"]-driver_before["fallback_3_calls"]==2 && warm["fallback_3_bytes"]-driver_before["fallback_3_bytes"]==32));
                std::printf("PATH initial XRE probes [4096,%d] host_calls=%llu (16B capability metadata)\n",rows,(unsigned long long)host);
                driver_before=warm;
            }
        }
        auto before=kunlun::statistics();
        auto invalid=*t[1];invalid.nb[1]+=4;ggml_tensor * bad[2]={t[0],&invalid};
        if(rows>1)REQUIRE(!sum(comm,bad));
        invalid=*t[1];invalid.type=GGML_TYPE_F16;REQUIRE(!sum(comm,bad));
        bad[1]=t[0];REQUIRE(!sum(comm,bad));REQUIRE(kunlun::statistics().peer_copies==before.peer_copies);
        auto driver_after=counters();
        REQUIRE(driver_after["sse_calls"]-driver_before["sse_calls"]==4);
        REQUIRE(driver_after["host_calls"]==driver_before["host_calls"]);
        std::printf("PATH warm SUM [4096,%d] sse=4 host=0 (2 all-reduces)\n",rows);
        for(auto b:buf)ggml_backend_buffer_free(b);
        ggml_free(c);
        std::printf("PASS SUM [4096,%d] bitwise CPU/rank equality guards repeat=3 negative=type/stride/owner\n",rows);
    }
    // Non-SSE-sized valid F32 payload uses the existing safe runtime fallback.
    {
        auto * c=context();ggml_tensor * t[2];ggml_backend_buffer_t buf[2];
        for(int d=0;d<2;++d) {t[d]=ggml_new_tensor_1d(c,GGML_TYPE_F32,3);buf[d]=ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backends[d]),256);REQUIRE(ggml_backend_tensor_alloc(buf[d],t[d],ggml_backend_buffer_get_base(buf[d]))==GGML_STATUS_SUCCESS);float v[3]={(float)d,2,3};ggml_backend_tensor_set(t[d],v,0,12);}
        REQUIRE(sum(comm,t));equal(get(t[0]),{1,4,6});equal(get(t[1]),{1,4,6});
        for(auto b:buf)ggml_backend_buffer_free(b);
        ggml_free(c);
        std::puts("PASS SUM odd-byte-size 12B safe existing fallback");
    }
    free_comm(comm);for(auto b:backends)ggml_backend_free(b);
}
static void graph_case(ggml_backend_t meta,int rows) {
    auto * c=context();auto * w=ggml_new_tensor_2d(c,GGML_TYPE_F32,64,4096);ggml_set_name(w,"weight");
    auto * x=ggml_new_tensor_2d(c,GGML_TYPE_F32,64,rows);ggml_set_name(x,"input");
    auto * bias=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,rows);ggml_set_name(bias,"bias");
    auto * storage=ggml_backend_alloc_ctx_tensors(c,meta);REQUIRE(storage);
    std::vector<float> hv(64*4096),hx(64*rows),hb(4096*rows,0.5f),ref(4096*rows);
    for(int n=0;n<4096;++n) for(int k=0;k<64;++k)hv[n*64+k]=(float)((k+n)%5-2)*0.25f;
    for(size_t i=0;i<hx.size();++i)hx[i]=(float)((int)i%7-3)*0.125f;
    for(int m=0;m<rows;++m)for(int n=0;n<4096;++n){float p[2]={};for(int d=0;d<2;++d)for(int k=d*32;k<(d+1)*32;++k)p[d]+=hv[n*64+k]*hx[m*64+k];ref[m*4096+n]=(p[0]+p[1])+0.5f;}
    ggml_backend_tensor_set(w,hv.data(),0,hv.size()*4);ggml_backend_tensor_set(x,hx.data(),0,hx.size()*4);ggml_backend_tensor_set(bias,hb.data(),0,hb.size()*4);
    auto * gc=context();auto * partial=ggml_mul_mat(gc,w,x);ggml_set_name(partial,"partial");
    auto * y=ggml_add(gc,partial,bias);ggml_set_name(y,"consumer");ggml_set_output(y);
    auto * graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,y);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(meta));REQUIRE(ggml_gallocr_alloc_graph(alloc,graph));
    ggml_backend_meta_split_state state;
    REQUIRE(ggml_backend_meta_tensor_split_state(partial,false,&state));REQUIRE(state.axis==GGML_BACKEND_SPLIT_AXIS_PARTIAL);
    REQUIRE(ggml_backend_meta_tensor_split_state(partial,true,&state));REQUIRE(state.axis==GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    REQUIRE(ggml_backend_supports_op(meta,partial));
    for(int rep=0;rep<3;++rep){auto before=kunlun::statistics();auto start=std::chrono::steady_clock::now();REQUIRE(ggml_backend_graph_compute(meta,graph)==GGML_STATUS_SUCCESS);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();std::printf("TIME graph M%d rep%d %.6f ms\n",rows,rep,ms);REQUIRE(kunlun::statistics().peer_copies-before.peer_copies==2);equal(get(y),ref);for(int d=0;d<2;++d)equal(get(ggml_backend_meta_tensor_child(y,d)),ref);}
    if(rows==1) {
        auto reg=ggml_backend_kunlun_reg();
        auto hook=(ggml_backend_meta_compute_group_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_meta_compute_group");
        if(hook && std::strcmp(std::getenv("GGML_KUNLUN_TENSOR_PARALLEL"),"0")) {
            ggml_backend_t bs[2]={ggml_backend_dev_init(ggml_backend_dev_by_name("KUNLUN0"),nullptr),ggml_backend_dev_init(ggml_backend_dev_by_name("KUNLUN1"),nullptr)};
            auto init=(ggml_backend_comm_init_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_init");
            auto fini=(ggml_backend_comm_free_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_free");
            void * comm=init(bs,2);REQUIRE(comm);
            auto invalid=*ggml_backend_meta_tensor_child(partial,0);
            invalid.src[0]=ggml_backend_meta_tensor_child(w,1);
            ggml_tensor * children[2]={&invalid,ggml_backend_meta_tensor_child(partial,1)};
            ggml_cgraph * gs[2]={ggml_new_graph(gc),ggml_new_graph(gc)};
            for(int d=0;d<2;++d){gs[d]->n_nodes=1;gs[d]->nodes[0]=children[d];}
            ggml_backend_meta_node_descriptor desc={partial,children,&state};
            auto before=kunlun::statistics();REQUIRE(hook(comm,gs,&desc,1,123)==GGML_STATUS_FAILED);
            state.axis=GGML_BACKEND_SPLIT_AXIS_UNKNOWN;invalid=*ggml_backend_meta_tensor_child(partial,0);
            REQUIRE(hook(comm,gs,&desc,1,124)==GGML_STATUS_FAILED);
            state.axis=GGML_BACKEND_SPLIT_AXIS_PARTIAL;invalid.op=GGML_OP_VIEW;invalid.type=GGML_TYPE_F16;
            REQUIRE(hook(comm,gs,&desc,1,125)==GGML_STATUS_FAILED);
            REQUIRE(kunlun::statistics().matmuls[0]==before.matmuls[0] && kunlun::statistics().matmuls[1]==before.matmuls[1] && kunlun::statistics().peer_copies==before.peer_copies);
            fini(comm);for(auto b:bs)ggml_backend_free(b);
            std::puts("PASS group rejects foreign producer/unknown state/unsupported PARTIAL before launch");
        }
    }
    // supports_op is shape-only at the simple backend; group preflight enforces ownership.
    auto transpose=ggml_transpose(gc,w);auto badop=*partial;badop.src[0]=transpose;
    REQUIRE(!ggml_backend_dev_supports_op(ggml_backend_dev_by_name("KUNLUN0"),&badop));
    ggml_gallocr_free(alloc);ggml_free(gc);ggml_backend_buffer_free(storage);ggml_free(c);
    std::printf("PASS Meta graph K64 N4096 M%d PARTIAL -> SUM -> ADD repeated=3\n",rows);
}
static void view_case(ggml_backend_t meta) {
    auto * c=context();auto * t=ggml_new_tensor_2d(c,GGML_TYPE_F32,64,8);ggml_set_name(t,"axis0");
    auto * buf=ggml_backend_alloc_ctx_tensors(c,meta);REQUIRE(buf);std::vector<float> v(512);for(int i=0;i<512;++i)v[i]=i;
    ggml_backend_tensor_set(t,v.data(),0,2048);
    auto * gc=context();auto * view=ggml_view_2d(gc,t,64,2,256,2*256);REQUIRE(ggml_backend_view_init(view)==GGML_STATUS_SUCCESS);
    for(int d=0;d<2;++d){auto * child=ggml_backend_meta_tensor_child(view,d);REQUIRE(child && child->view_offs==2*128);auto a=get(child);for(int row=0;row<2;++row)for(int k=0;k<32;++k)REQUIRE(a[row*32+k]==v[(row+2)*64+d*32+k]);}
    auto * tr=ggml_transpose(gc,t);REQUIRE(ggml_backend_view_init(tr)==GGML_STATUS_SUCCESS);
    REQUIRE(!ggml_is_contiguous(ggml_backend_meta_tensor_child(tr,0)));
    ggml_backend_meta_split_state state;REQUIRE(ggml_backend_meta_tensor_split_state(tr,false,&state));REQUIRE(state.axis==GGML_BACKEND_SPLIT_AXIS_1);
    ggml_free(gc);ggml_backend_buffer_free(buf);ggml_free(c);std::puts("PASS split view offset/stride transpose metadata");
}
static void row_ops(ggml_backend_t meta) {
    auto * c=context();auto * w=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,4);ggml_set_name(w,"rows");
    auto * ids=ggml_new_tensor_1d(c,GGML_TYPE_I32,2);ggml_set_name(ids,"indices");
    auto * dst=ggml_new_tensor_2d(c,GGML_TYPE_F16,256,4);ggml_set_name(dst,"cache");
    auto * storage=ggml_backend_alloc_ctx_tensors(c,meta);REQUIRE(storage);
    std::vector<float> v(1024);for(int i=0;i<1024;++i)v[i]=i*0.25f;
    int32_t ix[2]={1,3};ggml_backend_tensor_set(w,v.data(),0,v.size()*4);ggml_backend_tensor_set(ids,ix,0,8);ggml_backend_tensor_memset(dst,0,0,2048);
    auto * gc=context();auto * rows=ggml_get_rows(gc,w,ids);ggml_set_name(rows,"get_rows");
    REQUIRE(ggml_backend_supports_op(meta,rows)); // Mirrored physical input probe before allocation.
    auto * update=ggml_set_rows(gc,dst,rows,ids);ggml_set_name(update,"set_rows");ggml_set_output(rows);
    auto * graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,update);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(meta));REQUIRE(ggml_gallocr_alloc_graph(alloc,graph));
    REQUIRE(ggml_backend_supports_op(meta,rows));REQUIRE(ggml_backend_supports_op(meta,update));
    REQUIRE(ggml_backend_graph_compute(meta,graph)==GGML_STATUS_SUCCESS);
    std::vector<float> ref;ref.insert(ref.end(),v.begin()+256,v.begin()+512);ref.insert(ref.end(),v.begin()+768,v.end());equal(get(rows),ref);
    for(int d=0;d<2;++d) {
        equal(get(ggml_backend_meta_tensor_child(rows,d)),ref);
        std::vector<ggml_fp16_t> actual(1024);ggml_backend_tensor_get(ggml_backend_meta_tensor_child(dst,d),actual.data(),0,2048);
        for(int r=0;r<4;++r)for(int k=0;k<256;++k)REQUIRE(ggml_fp16_to_fp32(actual[r*256+k])==((r==1 || r==3)?v[r*256+k]:0));
    }
    ggml_gallocr_free(alloc);ggml_free(gc);ggml_backend_buffer_free(storage);ggml_free(c);
    std::puts("PASS physical child probe GET_ROWS/SET_ROWS F32->F16 mirrored graph");
}
static void n_split_graph(ggml_backend_t meta) {
    auto * c=context();auto * w=ggml_new_tensor_2d(c,GGML_TYPE_F32,64,64);ggml_set_name(w,"axis1_weight");
    auto * x=ggml_new_tensor_2d(c,GGML_TYPE_F32,64,1);ggml_set_name(x,"mirrored_input");
    auto * storage=ggml_backend_alloc_ctx_tensors(c,meta);REQUIRE(storage);
    std::vector<float> a(4096),b(64,1.0f/64),ref(64);
    for(int n=0;n<64;++n){ref[n]=2*(n%5-2);for(int k=0;k<64;++k)a[n*64+k]=n%5-2;}
    ggml_backend_tensor_set(w,a.data(),0,a.size()*4);ggml_backend_tensor_set(x,b.data(),0,b.size()*4);
    auto * gc=context();auto * mm=ggml_mul_mat(gc,w,x);auto * y=ggml_scale(gc,mm,2);
    auto * graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,y);ggml_set_output(y);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(meta));REQUIRE(ggml_gallocr_alloc_graph(alloc,graph));
    ggml_backend_meta_split_state state;REQUIRE(ggml_backend_meta_tensor_split_state(mm,false,&state));REQUIRE(state.axis==GGML_BACKEND_SPLIT_AXIS_0);
    REQUIRE(ggml_backend_meta_tensor_split_state(y,false,&state));REQUIRE(state.axis==GGML_BACKEND_SPLIT_AXIS_0);
    auto before=kunlun::statistics();REQUIRE(ggml_backend_graph_compute(meta,graph)==GGML_STATUS_SUCCESS);equal(get(y),ref);REQUIRE(kunlun::statistics().peer_copies==before.peer_copies);
    ggml_gallocr_free(alloc);ggml_free(gc);ggml_backend_buffer_free(storage);ggml_free(c);
    std::puts("PASS AXIS1 weight + MIRRORED input -> AXIS0 matmul/scale, no collective");
}
int main(int argc,char ** argv) {
    bool generic=argc>1 && !std::strcmp(argv[1],"--generic");
    setenv("GGML_KUNLUN_TENSOR_PARALLEL",generic?"0":"1",1);
    auto reg=ggml_backend_kunlun_reg();REQUIRE(ggml_backend_reg_dev_count(reg)==2);
    ggml_backend_dev_t devs[2]={ggml_backend_reg_dev_get(reg,0),ggml_backend_reg_dev_get(reg,1)};
    auto meta_dev=ggml_backend_meta_device(devs,2,policy,nullptr);REQUIRE(meta_dev);
    size_t before[2]={kunlun::allocated_bytes(0),kunlun::allocated_bytes(1)};
    auto meta=ggml_backend_dev_init(meta_dev,nullptr);REQUIRE(meta);
    placement(meta);view_case(meta);
    if(!generic)reduction(reg);
    graph_case(meta,1);graph_case(meta,128);
    n_split_graph(meta);
    if(!generic)row_ops(meta);
    ggml_backend_free(meta);
    REQUIRE(kunlun::allocated_bytes(0)==before[0] && kunlun::allocated_bytes(1)==before[1]);
    std::puts("PASS Phase1 complete; tracked HBM returned to baseline");
}
