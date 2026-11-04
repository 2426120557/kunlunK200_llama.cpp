#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Benchmark only. Apply after workers and warmup, outside timed work.
class bench_main_affinity {
public:
    bench_main_affinity() = default;
    bench_main_affinity(const bench_main_affinity &) = delete;
    bench_main_affinity & operator=(const bench_main_affinity &) = delete;
    ~bench_main_affinity() {
        if (active && !restore()) {
            std::abort();
        }
    }

    bool apply_from_env() {
        const char * value = std::getenv("LLAMA_BENCH_MAIN_CPU");
        if (!value || !*value || !std::strcmp(value, "off") || !std::strcmp(value, "none")) {
            return true;
        }
#if defined(__linux__)
        if (active || syscall(SYS_gettid) != getpid()) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR expected an inactive main-thread scope\n");
            return false;
        }
        errno = 0;
        char * end = nullptr;
        const long requested = std::strtol(value, &end, 10);
        if (errno || end == value || *end || *value < '0' || *value > '9' || requested < 0 || requested >= CPU_SETSIZE) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR invalid CPU; use a CPU number, or bench_affinity.py for auto\n");
            return false;
        }
        CPU_ZERO(&saved);
        if (sched_getaffinity(0, sizeof(saved), &saved) || !CPU_ISSET(requested, &saved)) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR CPU is outside the original allowed mask (errno=%d)\n", errno);
            return false;
        }
        dump_topology();
        std::fprintf(stderr, "BENCH_AFFINITY_BEFORE tid=%ld mask=%s cpu=%d target=%ld\n", syscall(SYS_gettid), mask_string(saved).c_str(), sched_getcpu(), requested);
        cpu_set_t target;
        CPU_ZERO(&target);
        CPU_SET(requested, &target);
        if (sched_setaffinity(0, sizeof(target), &target)) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR set failed errno=%d\n", errno);
            return false;
        }
        active = true;
        target_cpu = static_cast<int>(requested);
        cpu_set_t actual;
        CPU_ZERO(&actual);
        if (sched_getaffinity(0, sizeof(actual), &actual) || !CPU_EQUAL(&actual, &target)) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR applied mask mismatch\n");
            restore();
            return false;
        }
        std::fprintf(stderr, "BENCH_AFFINITY_APPLIED tid=%ld mask=%s cpu=%d\n", syscall(SYS_gettid), mask_string(actual).c_str(), sched_getcpu());
        return true;
#else
        std::fprintf(stderr, "BENCH_AFFINITY_ERROR main-thread affinity is supported on Linux only\n");
        return false;
#endif
    }

    bool restore() {
#if defined(__linux__)
        if (!active) {
            return true;
        }
        if (syscall(SYS_gettid) != getpid()) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR restore must run on the main thread\n");
            return false;
        }
        cpu_set_t measured;
        CPU_ZERO(&measured);
        const bool stable = sched_getaffinity(0, sizeof(measured), &measured) == 0 &&
                            CPU_COUNT(&measured) == 1 && CPU_ISSET(target_cpu, &measured);
        std::fprintf(stderr, "BENCH_AFFINITY_MEASURED_END tid=%ld mask=%s cpu=%d target=%d stable=%d\n", syscall(SYS_gettid), mask_string(measured).c_str(), sched_getcpu(), target_cpu, stable);
        if (sched_setaffinity(0, sizeof(saved), &saved)) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR restore failed errno=%d\n", errno);
            return false;
        }
        cpu_set_t actual;
        CPU_ZERO(&actual);
        if (sched_getaffinity(0, sizeof(actual), &actual) || !CPU_EQUAL(&actual, &saved)) {
            std::fprintf(stderr, "BENCH_AFFINITY_ERROR restored mask mismatch\n");
            return false;
        }
        active = false;
        std::fprintf(stderr, "BENCH_AFFINITY_RESTORED tid=%ld mask=%s cpu=%d\n", syscall(SYS_gettid), mask_string(actual).c_str(), sched_getcpu());
        return stable;
#endif
        return true;
    }

private:
    bool active = false;
#if defined(__linux__)
    cpu_set_t saved{};
    int target_cpu = -1;

    static std::string mask_string(const cpu_set_t & mask) {
        std::string result;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) {
                if (!result.empty()) result += ',';
                result += std::to_string(cpu);
            }
        }
        return result;
    }

    static std::string read(const std::string & path) {
        std::ifstream stream(path);
        std::string line;
        std::getline(stream, line);
        return line.empty() ? "unavailable" : line;
    }

    static void dump_topology() {
        const std::string root = "/sys/devices/system/cpu/";
        std::fprintf(stderr, "BENCH_AFFINITY_ONLINE cpus=%s\n", read(root + "online").c_str());
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            std::string base = root + "cpu" + std::to_string(cpu) + "/topology/";
            std::string core = read(base + "core_id");
            if (core == "unavailable") continue;
            std::fprintf(stderr, "BENCH_AFFINITY_TOPOLOGY cpu=%d package=%s core=%s siblings=%s\n", cpu,
                         read(base + "physical_package_id").c_str(), core.c_str(), read(base + "thread_siblings_list").c_str());
        }
        std::ifstream interrupts("/proc/interrupts");
        std::string line;
        bool found = false;
        while (std::getline(interrupts, line)) {
            if ((line.find("_dev") == std::string::npos || line.find("kl") == std::string::npos) &&
                line.find("xpu") == std::string::npos && line.find("XPU") == std::string::npos &&
                line.find("kunlun") == std::string::npos && line.find("Kunlun") == std::string::npos) continue;
            std::istringstream input(line);
            int irq;
            char colon;
            if (!(input >> irq >> colon) || colon != ':' || irq < 0) continue;
            found = true;
            std::string base = "/proc/irq/" + std::to_string(irq) + '/';
            std::fprintf(stderr, "BENCH_AFFINITY_IRQ irq=%d effective=%s requested=%s line=%s\n", irq,
                         read(base + "effective_affinity_list").c_str(), read(base + "smp_affinity_list").c_str(), line.c_str());
        }
        if (!found) std::fprintf(stderr, "BENCH_AFFINITY_IRQ unavailable=no_identified_xpu_irq\n");
    }
#endif
};
