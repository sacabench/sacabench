/*******************************************************************************
 * sacabench/util/span.hpp
 *
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <array>

#include <util/assertions.hpp>

// Inspired by the span type in https://github.com/Microsoft/GSL
template < typename T >
class span {
private:
    T* m_ptr;
    size_t m_size;

public:
    static constexpr size_t npos = -1ll;

    inline constexpr span() : m_ptr((T*)""), m_size(0) {
    }

    inline constexpr span(T* ptr, size_t size) : m_ptr(ptr), m_size(size) {
    }

    inline constexpr span(std::vector<T>& x) : span(x.data(), x.size()) {
    }

    // Iterators
    inline constexpr T* begin() const noexcept {
        return data();
    }

    inline constexpr T* end() const noexcept {
        return data() + size();
    }

    // Capacity
    inline constexpr size_t size() const noexcept {
        return m_size;
    }

    inline constexpr bool empty() const noexcept {
        return size() == 0;
    }

    // Element access
    inline constexpr T& operator[](size_t n) const {
        DCHECK_MSG(n >= 0, "Call of [] with n < 0");
        DCHECK_MSG(n < size(), "Call of [] with n >= size()");
        return *(data() + n);
    }

    inline constexpr T& at(size_t n) const {
        DCHECK_MSG(n >= 0, "Call of at with n < 0");
        DCHECK_MSG(n < size(), "Call of at with n >= size()");
        return *(data() + n);
    }

    inline constexpr T& front() const {
        DCHECK_MSG(size() != 0, "Call of front() with size() == 0");
        return *data();
    }

    inline constexpr T& back() const {
        DCHECK_MSG(size() != 0, "Call of back() with size() == 0");
        return *(data() + size() - 1);
    }

    inline constexpr T* data() const noexcept {
        return m_ptr;
    }

    // Modifiers
    inline constexpr void fill(const T& val = T()) const {
        for (auto& e : *this) {
            e = val;
        }
    }

    inline constexpr operator span< T const >() const {
        return span< T const >(data(), size());
    }

    inline constexpr span< T > slice(size_t from = 0, size_t to = npos) const {
        if (to == npos) {
            to = size();
        }
        DCHECK_MSG(0 <= from && from <= to && to <= size(), "Call of slice() with out-of bound values");
        return span< T >(data() + from, to - from);
    }
};
