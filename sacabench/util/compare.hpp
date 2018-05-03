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
    /// Adapter to turn a less-than comparison into an STL-conform
    /// greater-than binary predicate.
    template<typename compare_type>
    struct as_greater {
        as_greater(compare_type functor): less(functor) {}

        template<typename element_t>
        bool operator()(element_t const& lhs, element_t const& rhs) {
            return less(rhs, lhs);
        }
    private:
        compare_type less;
    };

    /// Adapter to turn a less-than comparison into an STL-conform
    /// equal binary predicate.
    template<typename compare_type>
    struct as_equal {
        as_equal(compare_type functor): less(functor) {}

        template<typename element_t>
        bool operator()(element_t const& lhs, element_t const& rhs) {
            return !less(lhs, rhs) && !less(rhs, lhs);
        }
    private:
        compare_type less;
    };

    /// Adapter to turn a less-than comparison into an STL-conform
    /// less-equal binary predicate.
    template<typename compare_type>
    struct as_less_equal {
        as_less_equal(compare_type functor): less(functor) {}

        template<typename element_t>
        bool operator()(element_t const& lhs, element_t const& rhs) {
            return !less(rhs, lhs);
        }
    private:
        compare_type less;
    };

    /// Adapter to turn a less-than comparison into an STL-conform
    /// greater-equal binary predicate.
    template<typename compare_type>
    struct as_greater_equal {
        as_greater_equal(compare_type functor): less(functor) {}

        template<typename element_t>
        bool operator()(element_t const& lhs, element_t const& rhs) {
            return !less(lhs, rhs);
        }
    private:
        compare_type less;
    };

    /// Adapter to turn a less-than comparison into an STL-conform
    /// not-equal binary predicate.
    template<typename compare_type>
    struct as_not_equal {
        as_not_equal(compare_type functor): less(functor) {}

        template<typename element_t>
        bool operator()(element_t const& lhs, element_t const& rhs) {
            return less(lhs, rhs) || less(rhs, lhs);
        }
    private:
        compare_type less;
    };
}

/******************************************************************************/
