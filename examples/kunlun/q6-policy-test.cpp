#include <initializer_list>
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <cstdio>
int main(){ggml_backend_load_all();auto*d=ggml_backend_dev_by_name("KUNLUN0");GGML_ASSERT(d);auto*c=ggml_init({1024*1024,nullptr,true});auto*w=ggml_new_tensor_2d(c,GGML_TYPE_Q6_K,2048,32000);for(int m:{1,2,512,513}){auto*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,2048,m);auto*y=ggml_mul_mat(c,w,x);setenv("KUNLUN_Q6_K_SDNN","0",1);GGML_ASSERT(!ggml_backend_dev_supports_op(d,y));setenv("KUNLUN_Q6_K_SDNN","1",1);GGML_ASSERT(ggml_backend_dev_supports_op(d,y)==(m<=512));}auto*bad=ggml_new_tensor_2d(c,GGML_TYPE_Q6_K,2048,32032);auto*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,2048,1);GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_mul_mat(c,bad,x)));auto*view=ggml_view_2d(c,w,2048,32000,w->nb[1],0);GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_mul_mat(c,view,x)));auto*half_x=ggml_new_tensor_2d(c,GGML_TYPE_F16,2048,1);GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_mul_mat(c,w,half_x)));unsetenv("KUNLUN_Q6_K_SDNN");GGML_ASSERT(!ggml_backend_dev_supports_op(d,ggml_mul_mat(c,w,x)));puts("FALLBACK_POLICY_PASS");ggml_free(c);}
