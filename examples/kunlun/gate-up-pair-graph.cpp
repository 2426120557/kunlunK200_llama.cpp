#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
static void ensure(bool v,const char*s){if(!v)throw std::runtime_error(s);}
static std::vector<unsigned char>get(ggml_tensor*t){std::vector<unsigned char>b(ggml_nbytes(t));ggml_backend_tensor_get(t,b.data(),0,b.size());return b;}
static ggml_cgraph slice(ggml_cgraph*g,int first,int end){auto v=*g;v.n_nodes=end-first;v.nodes=g->nodes+first;v.n_leafs=0;v.leafs=nullptr;v.grads=nullptr;v.grad_accs=nullptr;v.uid=0;return v;}
int main(int argc,char**argv){try{
 ensure(argc==5,"PD K N case");int pd=std::stoi(argv[1]),k=std::stoi(argv[2]),n=std::stoi(argv[3]);std::string name=argv[4];int m=name=="pp"?128:1;if(name=="unsupported")n=256;
 ggml_backend_load_all();auto dev=ggml_backend_dev_by_name(pd?"KUNLUN1":"KUNLUN0");ensure(dev,"device");auto backend=ggml_backend_dev_init(dev,nullptr);ensure(backend,"backend");
 auto ctx=ggml_init({4*1024*1024,nullptr,true});ensure(ctx,"ctx");auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);ggml_set_name(x,"ffn_norm");auto*input=x;
 if(name=="view")input=ggml_view_2d(ctx,x,k,m,x->nb[1],0);
 auto*w0=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_0,k,n),*w1=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_0,k,n);ggml_set_name(w0,"blk.0.ffn_gate.weight");ggml_set_name(w1,"blk.0.ffn_up.weight");
 auto*a=ggml_mul_mat(ctx,w0,input),*b=ggml_mul_mat(ctx,w1,input);ggml_set_name(a,"ffn_gate");ggml_set_name(b,"ffn_up");auto*glu=ggml_swiglu_split(ctx,a,b);ggml_set_name(glu,"ffn_swiglu");
 auto*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,glu);auto*out=glu;
 if(name=="multi"){auto*extra=ggml_scale(ctx,a,.5f);out=ggml_add(ctx,glu,extra);ggml_build_forward_expand(graph,out);}
 if(name=="inputmulti"){auto*extra=ggml_mul_mat(ctx,w0,input);out=ggml_add(ctx,glu,extra);ggml_build_forward_expand(graph,out);}
 auto*buffer=ggml_backend_alloc_ctx_tensors_from_buft(ctx,ggml_backend_dev_buffer_type(dev));ensure(buffer,"allocation");if(name=="alias")b->data=a->data;
 for(int which=0;which<2;++which){auto*w=which?w1:w0;std::vector<unsigned char>raw(ggml_nbytes(w));for(int row=0;row<n;++row)for(int block=0;block<k/32;++block){auto*p=raw.data()+((size_t)row*k/32+block)*18;uint16_t h=0x2400;std::memcpy(p,&h,2);for(int j=0;j<16;++j)p[j+2]=(row*3+block*17+j*13+which*41)&255;}ggml_backend_tensor_set(w,raw.data(),0,raw.size());}
 auto compute=[&](){if(name=="split"){auto left=slice(graph,0,1),right=slice(graph,1,ggml_graph_n_nodes(graph));ensure(ggml_backend_graph_compute(backend,&left)==GGML_STATUS_SUCCESS,"split first");ensure(ggml_backend_graph_compute(backend,&right)==GGML_STATUS_SUCCESS,"split remainder");}else ensure(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph");};
 for(int epoch=0;epoch<2;++epoch){std::vector<float>v((size_t)k*m);for(size_t j=0;j<v.size();++j)v[j]=std::sin(j*.071f+epoch*.5f)*3;
  setenv("GGML_KUNLUN_Q4_GATE_UP_PAIR","0",1);ggml_backend_tensor_set(x,v.data(),0,ggml_nbytes(x));compute();auto gold=get(out),ga=get(a),gb=get(b);
  setenv("GGML_KUNLUN_Q4_GATE_UP_PAIR","1",1);ggml_backend_tensor_set(x,v.data(),0,ggml_nbytes(x));compute();ensure(get(out)==gold&&get(a)==ga&&get(b)==gb,"graph bitwise");}
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);std::printf("GRAPH_PASS pd=%d k=%d case=%s bitwise=1 epochs=2\n",pd,k,name.c_str());return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"GRAPH_FAIL %s\n",e.what());return 1;}}
