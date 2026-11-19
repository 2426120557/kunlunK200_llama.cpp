static void rejected_attention(LayerGraph & graph) {
    auto reg=ggml_backend_kunlun_reg();auto init=(ggml_backend_comm_init_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_init");auto fini=(ggml_backend_comm_free_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_comm_free");auto hook=(ggml_backend_meta_compute_group_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_meta_compute_group");REQUIRE(init && fini && hook);
    ggml_backend_t bs[2]={ggml_backend_meta_simple_backend(graph.r.backend,0),ggml_backend_meta_simple_backend(graph.r.backend,1)};void * group=init(bs,2);REQUIRE(group);
    auto * logical=graph.stages[0].at("flash");ggml_backend_meta_split_state state{};REQUIRE(ggml_backend_meta_tensor_split_state(logical,false,&state));
    ggml_tensor invalid=*ggml_backend_meta_tensor_child(logical,0),query=*invalid.src[0];query.ne[1]=2;invalid.ne[2]=2;invalid.src[0]=&query;
    ggml_tensor * children[2]={&invalid,ggml_backend_meta_tensor_child(logical,1)};ggml_cgraph * gs[2]={ggml_new_graph(graph.c),ggml_new_graph(graph.c)};
    for(int d=0;d<2;++d){gs[d]->n_nodes=1;gs[d]->nodes[0]=children[d];}
    ggml_backend_meta_node_descriptor desc={logical,children,&state};auto stats=kunlun::statistics();auto memory=hbm();
    REQUIRE(hook(group,gs,&desc,1,8001)==GGML_STATUS_FAILED);
    invalid=*ggml_backend_meta_tensor_child(logical,1);REQUIRE(hook(group,gs,&desc,1,8002)==GGML_STATUS_FAILED);
    invalid=*ggml_backend_meta_tensor_child(logical,0);invalid.data=(char*)invalid.data+ggml_backend_buffer_get_size(invalid.buffer);REQUIRE(hook(group,gs,&desc,1,8003)==GGML_STATUS_FAILED);
    auto after=kunlun::statistics();REQUIRE(after.matmuls[0]==stats.matmuls[0] && after.matmuls[1]==stats.matmuls[1] && after.peer_copies==stats.peer_copies && hbm()==memory);fini(group);
    std::puts("PASS group preflight rejects unsupported query count / foreign owner / out-of-range view before launch");
}
