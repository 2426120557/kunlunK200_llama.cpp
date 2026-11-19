#pragma once
// Observational hooks only. No synchronization; unused without the audit preload.
#if defined(K200_DIAGNOSTIC_HOOKS) || defined(K200_COMPUTE_HOOKS)
#if !defined(__linux__) || !defined(__GNUC__)
#error K200 diagnostic hooks currently require Linux and GCC-compatible weak symbols.
#endif
#include <stdint.h>
#include <time.h>
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((weak,visibility("default"))) void k200_diag_event(const char *,const char *,const char *,int,int,uint64_t,uint64_t,uint64_t,uint64_t);
__attribute__((weak,visibility("default"))) int k200_diag_interest(const char *,const char *);
static inline uint64_t k200_diag_clock(clockid_t id) { struct timespec t;clock_gettime(id,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
#ifdef __cplusplus
}
struct K200DiagScope {
 const char *category,*name,*tensor;uint64_t start=0,cpu=0,bytes;int pd;
 K200DiagScope(const char *c,const char *n,const char *t="",uint64_t b=0,int device=-1):category(c),name(n),tensor(t),bytes(b),pd(device) {
  if(k200_diag_event){
   const int interest=k200_diag_interest?k200_diag_interest(c,n):2;
   if(interest){start=k200_diag_clock(CLOCK_MONOTONIC);if(interest==2)cpu=k200_diag_clock(CLOCK_THREAD_CPUTIME_ID);}
  }
 }
 ~K200DiagScope(){if(start)k200_diag_event(category,name,tensor,pd,1,start,k200_diag_clock(CLOCK_MONOTONIC),cpu?k200_diag_clock(CLOCK_THREAD_CPUTIME_ID)-cpu:0,bytes);}
};
#define K200_DIAG_SCOPE(variable,...) K200DiagScope variable(__VA_ARGS__)
#endif
#else
#define K200_DIAG_SCOPE(variable,...) ((void)0)
#endif
