/*******************************************************************************
 * https://github.com/tlx/tlx/blob/master/tlx/define/attribute_packed.hpp
 * https://github.com/tlx/tlx/blob/master/tlx/define/likely.hpp
 *
 * Part of tlx - http://panthema.net/tlx
 *
 * Copyright (C) 2015-2017 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the Boost Software License, Version 1.0
 ******************************************************************************/

#pragma once

namespace sacabench::util {

//! \addtogroup tlx_define
//! \{

/******************************************************************************/
// __attribute__ ((packed))

#if defined(__GNUC__) || defined(__clang__)
#define SB_ATTRIBUTE_PACKED __attribute__ ((packed))
#else
#define SB_ATTRIBUTE_PACKED
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SB_LIKELY(c) __builtin_expect((c), 1)
#define SB_UNLIKELY(c) __builtin_expect((c), 0)
#else
#define SB_LIKELY(c) c
#define SB_UNLIKELY(c) c
#endif

/// Prints current file and line number
#define TRACE() std::cerr << "TRACE @" << __FILE__ << ":" << __LINE__ << "\n";

// code compiled only in debug build (set build type to Debug)
#ifdef DEBUG
    /// `x` is compiled only in debug builds.
    #define IF_DEBUG(x) x
#else
    /// `x` is compiled only in debug builds.
    #define IF_DEBUG(x)
#endif

#define _GENSYM2(x,y) x##y
#define _GENSYM1(x,y) _GENSYM2(x,y)
// Generate a unique identifier
#define GENSYM(x) _GENSYM1(x,__COUNTER__)

//! \}

#define SB_FORCE_INLINE __attribute__((always_inline))
#define SB_NO_INLINE __attribute__((noinline))
#define SB_UNROLL_LOOPS __attribute__((optimize("unroll-loops")))

} // namespace util
