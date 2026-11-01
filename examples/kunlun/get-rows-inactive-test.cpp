#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(){
 setenv("GGML_KUNLUN_COMMON_Q4","1",1);setenv("GGML_KUNLUN_GET_ROWS","1",1);ggml_backend_load_all();auto*d=ggml_backend_dev_by_name("KUNLUN0");GGML_ASSERT(d);auto*backend=ggml_backend_dev_init(d,nullptr);GGML_ASSERT(backend);auto*c=ggml_init({1024*1024,nullptr,true});auto*w=ggml_new_tensor_2d(c,GGML_TYPE_Q4_0,64,32);auto*good=ggml_new_tensor_1d(c,GGML_TYPE_I32,1);auto*bad=ggml_new_tensor_1d(c,GGML_TYPE_I32,1);auto*a=ggml_get_rows(c,w,good);auto*b=ggml_get_rows(c,w,bad);auto*graph=ggml_new_graph(c);ggml_build_forward_order(graph,a);ggml_build_forward_order(graph,b);ggml_tensor*choices[]={a,b};auto*selected=ggml_build_forward_select(graph,choices,2,0);(void)selected;GGML_ASSERT(a->flags&GGML_TENSOR_FLAG_COMPUTE);GGML_ASSERT(!(b->flags&GGML_TENSOR_FLAG_COMPUTE));
 auto*buffer=ggml_backend_alloc_ctx_tensors(c,backend);GGML_ASSERT(buffer);std::vector<unsigned char>raw(32*36,0x88);for(int r=0;r<32;++r)for(int block=0;block<2;++block){raw[(r*2+block)*18]=0;raw[(r*2+block)*18+1]=0x3c;}ggml_backend_tensor_set(w,raw.data(),0,raw.size());int zero=0,invalid=-1;ggml_backend_tensor_set(good,&zero,0,4);ggml_backend_tensor_set(bad,&invalid,0,4);std::vector<float>sentinel(64,-12345.f),out(64);ggml_backend_tensor_set(b,sentinel.data(),0,256);GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(a,out.data(),0,256);for(float v:out)GGML_ASSERT(v==0);ggml_backend_tensor_get(b,out.data(),0,256);GGML_ASSERT(out==sentinel);puts("INACTIVE_GET_ROWS_SKIPPED_PASS");ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(c);
}
