
#ifndef _PARALLEL_H
#define _PARALLEL_H

// Settings are passed in from CMake.
#cmakedefine CILK
#cmakedefine CILKP
#cmakedefine OPENMP

// openmp
#if defined(OPENMP)
#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_main main
#define parallel_for _Pragma("omp parallel for") for
#define parallel_for_1 _Pragma("omp parallel for schedule (static,1)") for
#define parallel_for_256 _Pragma("omp parallel for schedule (static,256)") for

static int getWorkers() { return omp_get_max_threads(); }
static void setWorkers(int n) { omp_set_num_threads(n); }

// c++
#else
#define cilk_spawn
#define cilk_sync
#define parallel_main main
#define parallel_for for
#define parallel_for_1 for
#define parallel_for_256 for
#define cilk_for for

static int getWorkers() { return 1; }
static void setWorkers(int n) { }

#endif

#include <limits.h>

//#if defined(LONG)
typedef long intT;
typedef unsigned long uintT;
#define INT_T_MAX LONG_MAX
#define UINT_T_MAX ULONG_MAX
//#else
//typedef int intT;
//typedef unsigned int uintT;
//#define INT_T_MAX INT_MAX
//#define UINT_T_MAX UINT_MAX
//#endif

#endif // _PARALLEL_H
