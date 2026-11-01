#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc,char **argv) {
    setenv("GGML_KUNLUN_GET_ROWS_F32","1",1);ggml_backend_load_all();
    auto *d=ggml_backend_dev_by_name(argc>1&&std::atoi(argv[1])==1?"KUNLUN1":"KUNLUN0");GGML_ASSERT(d);
    auto *backend=ggml_backend_dev_init(d,nullptr);GGML_ASSERT(backend);
    auto *c=ggml_init({4*1024*1024,nullptr,true});
    auto *x=ggml_new_tensor_2d(c,GGML_TYPE_F32,2048,128);
    for(int m:{1,3,128,512,513}) {
        auto *i=ggml_new_tensor_1d(c,GGML_TYPE_I32,m);auto *y=ggml_get_rows(c,x,i);
        GGML_ASSERT(ggml_backend_dev_supports_op(d,y)==(m<=512));
        setenv("GGML_KUNLUN_GET_ROWS_F32","0",1);GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));
        unsetenv("GGML_KUNLUN_GET_ROWS_F32");GGML_ASSERT(ggml_backend_dev_supports_op(d,y)==(m<=512));
        setenv("GGML_KUNLUN_GET_ROWS_F32","invalid",1);GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));
        setenv("GGML_KUNLUN_GET_ROWS_F32","1",1);
    }
    auto *i=ggml_new_tensor_1d(c,GGML_TYPE_I32,3);auto *y=ggml_get_rows(c,x,i);
    setenv("GGML_KUNLUN_GET_ROWS","0",1);setenv("GGML_KUNLUN_COMMON_Q4","0",1);GGML_ASSERT(ggml_backend_dev_supports_op(d,y));
    auto saved=i->nb[0];i->nb[0]=8;GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));i->nb[0]=saved;
    y->nb[1]+=4;GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));y->nb[1]-=4;
    auto *v=ggml_view_2d(c,x,2048,128,x->nb[1],0);GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_get_rows(c,v,i)));
    auto *other=ggml_new_tensor_2d(c,GGML_TYPE_F16,2048,128);GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_get_rows(c,other,i)));
    using split_fn=ggml_backend_buffer_type_t(*)(int,const float*);
    auto split=(split_fn)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(d),"ggml_backend_split_buffer_type");GGML_ASSERT(split);
    float fractions[2]={.5f,.5f};auto *sb=ggml_backend_buft_alloc_buffer(split(0,fractions),0);x->buffer=sb;GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));x->buffer=nullptr;ggml_backend_buffer_free(sb);
    ggml_free(c);c=ggml_init({4*1024*1024,nullptr,true});
    x=ggml_new_tensor_2d(c,GGML_TYPE_F32,2048,128);auto *good=ggml_new_tensor_1d(c,GGML_TYPE_I32,3);auto *bad=ggml_new_tensor_1d(c,GGML_TYPE_I32,3);
    auto *a=ggml_get_rows(c,x,good);auto *b=ggml_get_rows(c,x,bad);auto *graph=ggml_new_graph(c);
    ggml_build_forward_order(graph,a);ggml_build_forward_order(graph,b);ggml_tensor *choices[]={a,b};ggml_build_forward_select(graph,choices,2,0);
    GGML_ASSERT(a->flags&GGML_TENSOR_FLAG_COMPUTE);GGML_ASSERT(!(b->flags&GGML_TENSOR_FLAG_COMPUTE));
    auto *buffer=ggml_backend_alloc_ctx_tensors(c,backend);GGML_ASSERT(buffer);
    std::vector<uint32_t> raw(2048*128),expected(2048*3),out(2048*3),sentinel(2048*3,0xdeadbeef);
    for(size_t j=0;j<raw.size();++j)raw[j]=(uint32_t)(j*1664525u+1013904223u);
    int ids[3]={127,0,127},invalid[3]={-1,128,-1};for(int j=0;j<3;++j)std::memcpy(expected.data()+j*2048,raw.data()+ids[j]*2048,2048*4);
    ggml_backend_tensor_set(x,raw.data(),0,raw.size()*4);ggml_backend_tensor_set(good,ids,0,12);ggml_backend_tensor_set(bad,invalid,0,12);ggml_backend_tensor_set(b,sentinel.data(),0,sentinel.size()*4);
    GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(a,out.data(),0,out.size()*4);GGML_ASSERT(out==expected);ggml_backend_tensor_get(b,out.data(),0,out.size()*4);GGML_ASSERT(out==sentinel);
    // Reuse the selected graph and buffers with new active indices at the same address.
    int next[3]={0,127,0};ggml_backend_tensor_set(good,next,0,12);ggml_backend_tensor_set(a,sentinel.data(),0,sentinel.size()*4);
    GGML_ASSERT(a->flags&GGML_TENSOR_FLAG_COMPUTE);GGML_ASSERT(!(b->flags&GGML_TENSOR_FLAG_COMPUTE));
    for(int j=0;j<3;++j)std::memcpy(expected.data()+j*2048,raw.data()+next[j]*2048,2048*4);
    GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(a,out.data(),0,out.size()*4);GGML_ASSERT(out==expected);ggml_backend_tensor_get(b,out.data(),0,out.size()*4);GGML_ASSERT(out==sentinel);
    puts("F32_GET_ROWS_BACKEND_POLICY_INACTIVE_PASS");ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(c);
}
