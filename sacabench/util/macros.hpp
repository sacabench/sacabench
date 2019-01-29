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

#ifdef DEBUG
#define SB_NOEXCEPT 
#else
#define SB_NOEXCEPT noexcept
#endif

// Include byteswap.h, if it is available. 
// On mac OS it is not available, rebuild it here.
// source: https://gist.github.com/atr000/249599
#if HAVE_BYTESWAP_H
#include <byteswap.h>
#else
#define bswap_16(value) \
((((value) & 0xff) << 8) | ((value) >> 8))

#define bswap_32(value) \
(((uint32_t)bswap_16((uint16_t)((value) & 0xffff)) << 16) | \
(uint32_t)bswap_16((uint16_t)((value) >> 16)))

#define bswap_64(value) \
(((uint64_t)bswap_32((uint32_t)((value) & 0xffffffff)) \
<< 32) | \
(uint64_t)bswap_32((uint32_t)((value) >> 32)))
#endif

} // namespace util
