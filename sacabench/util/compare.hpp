/*******************************************************************************
 * sacabench/util/compare.hpp
 *
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace sacabench::util {
/// Adapter to turn a less-than comparison into an
/// greater-than comparison.
template <typename compare_type>
struct as_greater {
    as_greater(compare_type functor) : less(functor) {}

    template <typename element_t>
    inline __attribute__((always_inline)) bool
    operator()(element_t const& lhs, element_t const& rhs) const {
        return less(rhs, lhs);
    }

private:
    compare_type less;
};

/// Adapter to turn a less-than comparison into an
/// equal comparison.
template <typename compare_type>
struct as_equal {
    as_equal(compare_type functor) : less(functor) {}

    template <typename element_t>
    inline __attribute__((always_inline)) bool
    operator()(element_t const& lhs, element_t const& rhs) const {
        return !less(lhs, rhs) && !less(rhs, lhs);
    }

private:
    compare_type less;
};

/// Adapter to turn a less-than comparison into an
/// less-equal comparison.
template <typename compare_type>
struct as_less_equal {
    as_less_equal(compare_type functor) : less(functor) {}

    template <typename element_t>
    inline __attribute__((always_inline)) bool
    operator()(element_t const& lhs, element_t const& rhs) const {
        return !less(rhs, lhs);
    }

private:
    compare_type less;
};

/// Adapter to turn a less-than comparison into an
/// greater-equal comparison.
template <typename compare_type>
struct as_greater_equal {
    as_greater_equal(compare_type functor) : less(functor) {}

    template <typename element_t>
    inline __attribute__((always_inline)) bool
    operator()(element_t const& lhs, element_t const& rhs) const {
        return !less(lhs, rhs);
    }

private:
    compare_type less;
};

/// Adapter to turn a less-than comparison into an
/// not-equal comparison.
template <typename compare_type>
struct as_not_equal {
    as_not_equal(compare_type functor) : less(functor) {}

    template <typename element_t>
    inline __attribute__((always_inline)) bool
    operator()(element_t const& lhs, element_t const& rhs) const {
        return less(lhs, rhs) || less(rhs, lhs);
    }

private:
    compare_type less;
};

/// Adapter that allows defining a compare
/// function by mapping the values into another value and comparing that.
///
/// Example: Sorting strings only according to their length.
/// ```cpp
/// auto by_size = compare_key([](auto const& s) { return s.size(); });
/// sort(list_of_strings, by_size);
/// ```
template <typename key_function, typename compare_function = std::less<void>>
struct compare_key {
    compare_key(key_function key, compare_function cmp = compare_function())
        : m_key(key), m_cmp(cmp) {}

    template <typename element_type>
    inline __attribute__((always_inline)) bool
    operator()(element_type const& lhs, element_type const& rhs) const {
        return m_cmp(m_key(lhs), m_key(rhs));
    }

private:
    key_function m_key;
    compare_function m_cmp;
};
} // namespace sacabench::util

/******************************************************************************/
