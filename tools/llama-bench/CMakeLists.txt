# llama-bench-impl: benchmark logic, reusable by app

set(TARGET llama-bench-impl)

add_library(${TARGET} llama-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
    install(FILES bench_affinity.py kunlun_guarded_runner.py DESTINATION share/llama-bench)
    install(PROGRAMS bench_affinity.py DESTINATION bin RENAME llama-bench-affinity)
    install(FILES bench-main-affinity.h DESTINATION include/llama-bench)
    install(FILES BENCH_AFFINITY.ja.md DESTINATION share/doc/llama-bench)
endif()
