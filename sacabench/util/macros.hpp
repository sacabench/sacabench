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

//! \}

} // namespace util
