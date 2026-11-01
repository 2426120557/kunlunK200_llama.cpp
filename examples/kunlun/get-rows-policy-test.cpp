#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
int main(){
 setenv("GGML_KUNLUN_COMMON_Q4","1",1);setenv("GGML_KUNLUN_GET_ROWS","1",1);ggml_backend_load_all();auto*d=ggml_backend_dev_by_name("KUNLUN0");GGML_ASSERT(d);auto*c=ggml_init({1024*1024,nullptr,true});auto*w=ggml_new_tensor_2d(c,GGML_TYPE_Q4_0,2048,32000);auto*normal=ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(d),0);w->buffer=normal;
 for(int m:{1,3,128,512,513}){auto*i=ggml_new_tensor_1d(c,GGML_TYPE_I32,m);auto*y=ggml_get_rows(c,w,i);GGML_ASSERT(ggml_backend_dev_supports_op(d,y)==(m<=512));setenv("GGML_KUNLUN_GET_ROWS","0",1);GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));unsetenv("GGML_KUNLUN_GET_ROWS");GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));setenv("GGML_KUNLUN_GET_ROWS","1",1);}
 auto*i=ggml_new_tensor_1d(c,GGML_TYPE_I32,128);auto*y=ggml_get_rows(c,w,i);auto saved=i->nb[0];i->nb[0]=8;GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));i->nb[0]=saved;y->nb[1]+=4;GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));y=ggml_get_rows(c,w,i);
 auto*v=ggml_view_2d(c,w,2048,32000,w->nb[1],0);GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_get_rows(c,v,i)));auto*other=ggml_new_tensor_2d(c,GGML_TYPE_F16,2048,32000);GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_get_rows(c,other,i)));
 setenv("GGML_KUNLUN_COMMON_Q4","0",1);GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));setenv("GGML_KUNLUN_COMMON_Q4","1",1);
 auto*cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);auto*cb=ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cpu),0);w->buffer=cb;GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));w->buffer=normal;ggml_backend_buffer_free(cb);
 using split_fn=ggml_backend_buffer_type_t(*)(int,const float*);auto split=(split_fn)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(d),"ggml_backend_split_buffer_type");GGML_ASSERT(split);float fractions[2]={.5f,.5f};auto*sb=ggml_backend_buft_alloc_buffer(split(0,fractions),0);w->buffer=sb;GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));w->buffer=normal;ggml_backend_buffer_free(sb);ggml_backend_buffer_free(normal);w->buffer=nullptr;ggml_free(c);puts("GET_ROWS_POLICY_PASS");
}
