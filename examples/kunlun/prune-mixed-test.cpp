#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
static void need(bool ok,const char *message) { if(!ok){std::fprintf(stderr,"HOST_TEST_ERROR %s\n",message);std::exit(2);} }
int main(int argc,char **argv) {
    need(argc==4,"model embedding_offset output");llama_backend_init();
    ggml_backend_dev_t devices[]={ggml_backend_dev_by_name("KUNLUN0"),nullptr};need(devices[0],"KUNLUN0");
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=99;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    auto *model=llama_model_load_from_file(argv[1],mp);need(model,"model load");
    auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=128;cp.n_ubatch=128;cp.n_threads=6;cp.n_threads_batch=6;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *ctx=llama_init_from_model(model,cp);need(ctx,"context");auto *batch=llama_batch_ext_init(ctx);need(batch,"batch");
    const int k=llama_model_n_embd(model),nv=llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto *file=std::fopen(argv[3],"wb");need(file,"output file");
    const char *mode=std::getenv("GGML_KUNLUN_PRUNE_INACTIVE_MIXED");const bool enabled=!mode||!std::strcmp(mode,"1");
    int pos=0;std::vector<float>embd(k);std::ifstream raw(argv[1],std::ios::binary);size_t offset=std::strtoull(argv[2],nullptr,10),rowbytes=(size_t)k/32*18;std::vector<unsigned char>row(rowbytes);
    auto embedding=[&](int id){raw.seekg(offset+(size_t)id*rowbytes);raw.read((char*)row.data(),rowbytes);need((size_t)raw.gcount()==rowbytes,"embedding bytes");for(int b=0;b<k/32;++b){uint16_t bits=row[b*18]|row[b*18+1]<<8;_Float16 h;std::memcpy(&h,&bits,2);float d=h;for(int j=0;j<16;++j){int q=row[b*18+2+j];embd[b*32+j]=float((q&15)-8)*d;embd[b*32+j+16]=float((q>>4)-8)*d;}}};
    auto process=[&](const char *label,int kind,int count,int first) {
        const char *flag=std::getenv("GGML_KUNLUN_PRUNE_INACTIVE_MIXED");
        std::fprintf(stderr,"PRUNE_CASE name=%s kind=%d count=%d pos=%d flag=%s\n",label,kind,count,pos,flag?flag:"unset");
        llama_batch_ext_clear(batch);
        for(int i=0;i<count;++i) {
            int slot;
            if(kind==1||(kind==2&&i%2)){embedding(first+i);slot=llama_batch_ext_add_embd(batch,0,{embd.data(),1,(size_t)k});}
            else slot=llama_batch_ext_add_token(batch,0,first+i);
            need(slot>=0,"add row");llama_pos p=pos+i;need(llama_batch_ext_set_pos(batch,slot,&p),"position");
        }
        need(llama_batch_ext_set_output_logits(batch,count-1,true),"output selection");need(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch)==0,"decode");pos+=count;
        auto *logits=llama_get_logits_ith(ctx,-1);need(logits,"logits");for(int i=0;i<nv;++i)need(std::isfinite(logits[i]),"nonfinite");
        need(std::fwrite(logits,4,nv,file)==(size_t)nv,"write logits");return std::vector<float>(logits,logits+nv);
    };
    process("token_pp",0,8,1);process("token_tg",0,1,17);process("updated_token",0,1,18);process("same_token",0,1,18);
    process("external_embedding",1,1,19);process("token_after_embedding",0,1,20);process("mixed_token_embedding",2,2,21);process("token_after_mixed",0,2,23);
    setenv("GGML_KUNLUN_PRUNE_INACTIVE_MIXED",enabled?"0":"1",1);process("flag_changed",0,2,25);
    setenv("GGML_KUNLUN_PRUNE_INACTIVE_MIXED",enabled?"1":"0",1);process("flag_restored",0,2,27);
    std::vector<unsigned char>state(llama_state_get_size(ctx));need(llama_state_get_data(ctx,state.data(),state.size())==state.size(),"save KV state");int saved_pos=pos;
    auto before=process("state_before",0,1,29);need(llama_state_set_data(ctx,state.data(),state.size())==state.size(),"restore KV state");pos=saved_pos;
    auto after=process("state_after",0,1,29);need(!std::memcmp(before.data(),after.data(),nv*4),"KV logits mismatch");
    std::fclose(file);llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();std::puts("PRUNE_INPUT_FORMS_KV_PASS");
}
