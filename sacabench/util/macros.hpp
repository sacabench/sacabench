/*******************************************************************************
 * tlx/define/attribute_packed.hpp
 * tlx/define/likely.hpp
 *
 * Part of tlx - http://panthema.net/tlx
 *
 * Copyright (C) 2015-2017 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the Boost Software License, Version 1.0
 ******************************************************************************/

#pragma once

namespace util {

//! \addtogroup tlx_define
//! \{

/******************************************************************************/
// __attribute__ ((packed))

#if defined(__GNUC__) || defined(__clang__)
#define TLX_ATTRIBUTE_PACKED __attribute__ ((packed))
#else
#define TLX_ATTRIBUTE_PACKED
#endif

//! \}

} // namespace tlx

namespace tlx {

//! \addtogroup tlx_define
//! \{

#if defined(__GNUC__) || defined(__clang__)
#define TLX_LIKELY(c)   __builtin_expect((c), 1)
#define TLX_UNLIKELY(c) __builtin_expect((c), 0)
#else
#define TLX_LIKELY(c)   c
#define TLX_UNLIKELY(c) c
#endif

//! \}

} // namespace util
