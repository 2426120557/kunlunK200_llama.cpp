template<class T> static std::vector<T> device_read(int d,const void * p,size_t count) {
    std::lock_guard<std::recursive_mutex> lock(kunlun::runtime_mutex());kunlun::select(d);kunlun::check(kunlun::wait());
    std::vector<T> v(count);kunlun::check(kunlun::copy(v.data(),p,count*sizeof(T),kunlun::device_to_host));kunlun::check(kunlun::wait());return v;
}
static void device_write(int d,void * p,const void * v,size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(kunlun::runtime_mutex());kunlun::select(d);kunlun::check(kunlun::copy(p,v,bytes,kunlun::host_to_device));kunlun::check(kunlun::wait());
}
static void verify_packed(Run & run) {
    using Query=kunlun::CommonQ4Info (*)(const ggml_tensor *);
    auto query=(Query)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_weight_info");REQUIRE(query);
    for(auto * logical:run.w)for(int d=0;d<2;++d){auto * t=ggml_backend_meta_tensor_child(logical,d);REQUIRE(t);auto a=query(t);
        REQUIRE(a.device==d && a.abi_version==1 && a.layout==1 && a.scale_order==1 && a.k==t->ne[0] && a.n==t->ne[1]);
        std::vector<unsigned char> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());
        auto packed=device_read<unsigned char>(d,t->data,a.payload_bytes);
        for(int col=0;col<a.np;++col)for(int kb=0;kb<a.kp/32;++kb){
            bool valid=col<a.n && kb<a.k/32;size_t si=a.scale_offset+((size_t)kb*a.np+col)*2;size_t src=((size_t)col*(a.k/32)+kb)*18;
            REQUIRE(packed[si]==(valid?raw[src]:0) && packed[si+1]==(valid?raw[src+1]:0));
            auto q=[&](int j){return valid?(((raw[src+2+j%16]>>(j/16*4))&15)-8)&15:0;};
            for(int p=0;p<16;++p){int row=(col%32/16)*8+p/2,bank=col%16;
                size_t wi=((size_t)(col/32)*(a.kp/32)+kb)*512+(bank*16+row)*2+p%2;
                REQUIRE(packed[wi]==(q(p*2)|(q(p*2+1)<<4)));
            }
        }
        std::printf("PASS packed scale/nibble/padding %s PD%d K=%d Kp=%d N=%d Np=%d\n",logical->name,d,a.k,a.kp,a.n,a.np);
    }
}
struct QuantCopy { int kp,m,components;std::vector<unsigned char> q,params; };
static QuantCopy snapshot(int d,kunlun::CommonQ4Quantization v) {
    REQUIRE(v.values && v.scales && v.rows>0 && v.kp>0);
    return {v.kp,v.rows,v.components,device_read<unsigned char>(d,v.values,v.components==2?4*v.kp+1024:v.rows*v.kp),device_read<unsigned char>(d,v.scales,v.components==2?64:v.rows*64)};
}
static void check_quant(const QuantCopy * local,const QuantCopy & full) {
    REQUIRE(local[0].kp==5632 && local[1].kp==5632 && full.kp==11008);
    for(int d=0;d<2;++d){auto & q=local[d];REQUIRE(q.m==full.m && q.components==full.components && q.params==full.params);
        if(q.components==2){
            for(int component=0;component<2;++component)for(int j=0;j<q.kp;++j){
                size_t index=component*q.kp*2+(j/32)*64+j%32;
                size_t global=d*5504+j;
                unsigned char expected=j<5504?full.q[component*full.kp*2+(global/32)*64+global%32]:0;
                REQUIRE(q.q[index]==expected);
            }
            for(int component=0;component<2;++component)for(int b=0;b<q.kp/32;++b)for(int j=32;j<64;++j)REQUIRE(q.q[component*q.kp*2+b*64+j]==0);
            for(int j=4*q.kp;j<4*q.kp+1024;++j)REQUIRE(q.q[j]==0);
        }else{
            for(int row=0;row<q.m;++row)for(int j=0;j<q.kp;++j)REQUIRE(q.q[row*q.kp+j]==(j<5504?full.q[row*full.kp+d*5504+j]:0));
        }
    }
    std::printf("PASS GLOBAL quant bytes/scales bitwise, zero padding M=%d components=%d\n",full.m,full.components);
}
static QuantCopy full_quant(const std::vector<float> & input,int m) {
    kunlun::CommonQ4Workspace workspace;kunlun::CommonQ4ActivationLease lease;kunlun::Buffer x(0,input.size()*4);
    device_write(0,x.data(),input.data(),input.size()*4);
    REQUIRE(kunlun::common_q4_prepare(workspace,0,x.data(),11008,m,kunlun::InputType::f32,999,lease));
    return snapshot(0,workspace.quantization());
}
static void verify_graph_quant(Run & run,Graph & graph) {
    using Query=kunlun::CommonQ4Quantization (*)(ggml_backend_t);
    auto query=(Query)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_quantization");REQUIRE(query);
    QuantCopy parts[2];for(int d=0;d<2;++d)parts[d]=snapshot(d,query(ggml_backend_meta_simple_backend(run.backend,d)));
    // Test-only gather, outside the execution/timing window.
    check_quant(parts,full_quant(read(graph.glu),run.rows));
}
static void quant_edges(int m) {
    const int count=5504*m;const size_t bytes=(size_t)count*4;
    kunlun::Buffer x0(0,bytes+512),x1(1,bytes+512);void * inputs[2]={(char *)x0.data()+256,(char *)x1.data()+256};
    std::vector<float> values[2]={std::vector<float>(count+128,9999),std::vector<float>(count+128,9999)},full(11008*m);
    for(int d=0;d<2;++d)for(int i=0;i<count;++i)values[d][i+64]=0;
    // Keep the first PP row all-zero; exercise each side of the split elsewhere.
    int row=m==1?0:1;values[0][64+row*5504+5503]=-2.0f;values[1][64+row*5504]=3.0f;
    for(int r=0;r<m;++r)for(int d=0;d<2;++d)std::memcpy(full.data()+r*11008+d*5504,values[d].data()+64+r*5504,5504*4);
    kunlun::CommonQ4Workspace w0,w1;kunlun::CommonQ4ActivationLease l0,l1;
    {
        std::vector<float> zeros[2]={std::vector<float>(count+128,9999),std::vector<float>(count+128,9999)};
        for(int d=0;d<2;++d){std::fill(zeros[d].begin()+64,zeros[d].end()-64,0);device_write(d,d?x1.data():x0.data(),zeros[d].data(),bytes+512);}
        REQUIRE(kunlun::common_q4_tp_prepare_global(w0,w1,inputs[0],inputs[1],5504,m,1000,l0,l1));
        QuantCopy zero[2]={snapshot(0,w0.quantization()),snapshot(1,w1.quantization())};check_quant(zero,full_quant(std::vector<float>(11008*m,0),m));
    }
    for(int d=0;d<2;++d)device_write(d,d?x1.data():x0.data(),values[d].data(),bytes+512);
    REQUIRE(kunlun::common_q4_tp_prepare_global(w0,w1,inputs[0],inputs[1],5504,m,1001,l0,l1));
    QuantCopy q[2]={snapshot(0,w0.quantization()),snapshot(1,w1.quantization())};check_quant(q,full_quant(full,m));
    for(int d=0;d<2;++d){auto actual=device_read<float>(d,d?x1.data():x0.data(),count+128);REQUIRE(actual==values[d]);}
    auto before=kunlun::statistics();REQUIRE(!kunlun::common_q4_tp_prepare_global(w0,w1,inputs[0],inputs[1],5503,m,1002,l0,l1));
    REQUIRE(!kunlun::common_q4_tp_prepare_global(w0,w1,(char *)inputs[0]+1,inputs[1],5504,m,1002,l0,l1));REQUIRE(kunlun::statistics().peer_copies==before.peer_copies);
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
        device_write(1,inputs[1],&bad,4);REQUIRE(!kunlun::common_q4_tp_prepare_global(w0,w1,inputs[0],inputs[1],5504,m,1003,l0,l1));
    }
    std::printf("PASS quant split-boundary/zero row/source guards/invalid shape/alignment/NaN/Inf M=%d\n",m);
}
static void partial_guards(Run & run,Graph & graph) {
    int m=run.rows;auto full=read(graph.glu);const size_t xs=(size_t)m*5504*4,ys=(size_t)m*4096*4;
    kunlun::CommonQ4Workspace w0,w1;kunlun::CommonQ4ActivationLease l0,l1;
    kunlun::Buffer x0(0,xs+512),x1(1,xs+512),y0(0,ys+512),y1(1,ys+512);
    void * x[2]={(char *)x0.data()+256,(char *)x1.data()+256};void * y[2]={(char *)y0.data()+256,(char *)y1.data()+256};
    std::vector<float> source[2]={std::vector<float>(xs/4+128,9999),std::vector<float>(xs/4+128,9999)},dest(ys/4+128,-12345);
    for(int d=0;d<2;++d){for(int row=0;row<m;++row)std::memcpy(source[d].data()+64+row*5504,full.data()+row*11008+d*5504,5504*4);device_write(d,d?x1.data():x0.data(),source[d].data(),xs+512);device_write(d,d?y1.data():y0.data(),dest.data(),ys+512);}
    REQUIRE(kunlun::common_q4_tp_prepare_global(w0,w1,x[0],x[1],5504,m,1005,l0,l1));
    using Handle=const kunlun::CommonQ4Weight *(*)(const ggml_tensor *);
    auto handle=(Handle)ggml_backend_reg_get_proc_address(ggml_backend_kunlun_reg(),"ggml_backend_kunlun_tp_weight_handle");REQUIRE(handle);
    kunlun::CommonQ4Workspace * work[2]={&w0,&w1};kunlun::CommonQ4ActivationLease * leases[2]={&l0,&l1};
    for(int d=0;d<2;++d){auto * weight=handle(ggml_backend_meta_tensor_child(run.w[2],d));REQUIRE(weight);REQUIRE(kunlun::common_q4_tp_submit(*work[d],d,y[d],*weight,*leases[d],1005));}
    for(int d=0;d<2;++d)kunlun::common_q4_finish_submit(*work[d]);
    for(int d=0;d<2;++d)kunlun::common_q4_complete(*work[d]);
    std::vector<float> values[2];
    for(int d=0;d<2;++d){auto actual=device_read<float>(d,d?y1.data():y0.data(),ys/4+128);for(int i=0;i<64;++i)REQUIRE(actual[i]==-12345 && actual[ys/4+64+i]==-12345);values[d]=std::vector<float>(actual.begin()+64,actual.end()-64);REQUIRE(device_read<float>(d,d?x1.data():x0.data(),xs/4+128)==source[d]);}
    for(size_t i=0;i<values[0].size();++i)values[0][i]+=values[1][i];compare("SUM vs explicit CPU P0+P1",read(ggml_backend_meta_tensor_child(graph.down,0)),values[0],1e-6);
    std::printf("PASS Down partial output guards and independent SUM reference M=%d\n",m);
}
static std::vector<float> cpu_matrix(const std::vector<unsigned char> & raw,int k,int n,const std::vector<float> & x) {
    REQUIRE(x.size()==(size_t)k);std::vector<float> y(n);
    for(int row=0;row<n;++row){double sum=0;
        for(int b=0;b<k/32;++b){const auto * p=raw.data()+((size_t)row*(k/32)+b)*18;ggml_fp16_t h;std::memcpy(&h,p,2);float scale=ggml_fp16_to_fp32(h);double dot=0;
            for(int j=0;j<32;++j)dot+=((int)((p[2+j%16]>>(j/16*4))&15)-8)*(double)x[b*32+j];sum+=dot*scale;}
        y[row]=(float)sum;
    }
    return y;
}
static void cpu_reference(ModelData & data,const std::vector<float> & input,Graph & graph) {
    auto gate=cpu_matrix(data.raw[0],4096,11008,input),up=cpu_matrix(data.raw[1],4096,11008,input);std::vector<float> glu(11008);
    compare("Gate vs CPU dequant/F64 accumulation",read(graph.gate),gate,0.005);
    compare("Up vs CPU dequant/F64 accumulation",read(graph.up),up,0.005);
    for(int i=0;i<11008;++i)glu[i]=(float)((double)gate[i]/(1.0+std::exp(-(double)gate[i]))*up[i]);
    auto down=cpu_matrix(data.raw[2],11008,4096,glu);compare("GLOBAL FFN vs CPU dequant/F64 reference",read(ggml_backend_meta_tensor_child(graph.down,0)),down,0.01);
}
