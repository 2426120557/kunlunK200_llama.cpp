#include "../bench-main-affinity.h"
#include <thread>
#include <atomic>
#include <cassert>
#include <chrono>

int main() {
    cpu_set_t original, worker_original, actual;
    assert(sched_getaffinity(0, sizeof(original), &original) == 0);
    int target = 0;
    while (target < CPU_SETSIZE && !CPU_ISSET(target, &original)) ++target;
    assert(target < CPU_SETSIZE);
    std::atomic<bool> ready{false}, done{false};
    std::thread worker([&]() {
        assert(sched_getaffinity(0, sizeof(worker_original), &worker_original) == 0);
        ready = true;
        while (!done) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    while (!ready) std::this_thread::yield();
    setenv("LLAMA_BENCH_MAIN_CPU", std::to_string(target).c_str(), 1);
    {
        bench_main_affinity scope;
        assert(scope.apply_from_env());
        assert(sched_getaffinity(0, sizeof(actual), &actual) == 0);
        assert(CPU_COUNT(&actual) == 1 && CPU_ISSET(target, &actual));
        assert(pthread_getaffinity_np(worker.native_handle(), sizeof(actual), &actual) == 0);
        assert(CPU_EQUAL(&actual, &worker_original));
        assert(scope.restore());
    }
    assert(sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_EQUAL(&actual, &original));
    try {
        bench_main_affinity scope;
        assert(scope.apply_from_env());
        throw 1;
    } catch (int) {}
    assert(sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_EQUAL(&actual, &original));
    if (CPU_COUNT(&original) > 1) {
        bench_main_affinity scope;
        assert(scope.apply_from_env());
        assert(sched_setaffinity(0, sizeof(original), &original) == 0);
        assert(!scope.restore());
    }
    assert(sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_EQUAL(&actual, &original));
    setenv("LLAMA_BENCH_MAIN_CPU", "1024", 1);
    { bench_main_affinity scope; assert(!scope.apply_from_env()); }
    assert(sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_EQUAL(&actual, &original));
    setenv("LLAMA_BENCH_MAIN_CPU", "off", 1);
    { bench_main_affinity scope; assert(scope.apply_from_env()); }
    assert(sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_EQUAL(&actual, &original));
    done = true;
    worker.join();
    std::puts("AFFINITY_THREAD_AND_RESTORE_PASS");
}
