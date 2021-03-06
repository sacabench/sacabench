/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "macros.hpp"
#include <util/assertions.hpp>

namespace sacabench::util {

/// A debug write function, that prints an integer to a ostream as
/// a decimal number, even if it is a char.
template<typename T>
inline std::ostream& write_map_char_to_int(T const& v, std::ostream& out) {
    return out << v;
}
template<>
inline std::ostream& write_map_char_to_int<unsigned char>(unsigned char const& v, std::ostream& out) {
    return out << uint64_t(v);
}
template<>
inline std::ostream& write_map_char_to_int<char>(char const& v, std::ostream& out) {
    return out << int64_t(v);
}
template<>
inline std::ostream& write_map_char_to_int<signed char>(signed char const& v, std::ostream& out) {
    return out << int64_t(v);
}

// Inspired by the span type in https://github.com/Microsoft/GSL
/// A wrapper around a (pointer, len) pair.
///
/// Allows easy access to a slice of memory.
template <typename T>
class span {
private:
    T* m_ptr;
    size_t m_size;

    // A debug check to catch invalidated spans
    IF_DEBUG(mutable std::shared_ptr<bool> m_alive_check;)
public:
    IF_DEBUG(inline void register_alive_check(std::shared_ptr<bool> ptr)
                 const { m_alive_check = ptr; })
private:
    inline void check_invalidated() const {
        IF_DEBUG(if (m_alive_check) {
            DCHECK_MSG(*m_alive_check,
                       "The container this span points at no longer exists.");
        })
    }

public:
    /// Special value that means "the end of the slice".
    static constexpr size_t npos = -1ll;

    /// Create a span of size 0.
    inline constexpr span() : m_ptr((T*)""), m_size(0) {}

    /// Create a span from an pointer and a length.
    inline span(T* ptr, size_t size) : m_ptr(ptr), m_size(size) {
        DCHECK_MSG(size * sizeof(T) < (1ull << 48),
                   "Trying to create a `span` with a size of "
                       << size
                       << ", which is unrealistic large. This likely happened "
                          "through a underflow "
                          "when calculating a size `< 0`.")
    }

    /// Constructor from a `std::vector`.
    inline constexpr span(std::vector<T>& x) : span(x.data(), x.size()) {}

    /// Constructor from a `std::array`.
    template <size_t N>
    inline constexpr span(std::array<T, N>& x) : span(x.data(), x.size()) {}

    /// Constructor from a std container.
    ///
    /// This means a type providing a `data()` and `size()` method.
    template <typename std_container_type>
    inline constexpr span(std_container_type& x) : span(x.data(), x.size()) {}

    // Iterators

    /// Iterator to the begin of the slice.
    inline T* begin() const {
        check_invalidated();
        return data();
    }

    /// Iterator to the end of the slice.
    inline T* end() const {
        check_invalidated();
        return data() + size();
    }

    // Capacity

    /// Size of the slice of memory, in elements.
    inline constexpr size_t size() const noexcept { return m_size; }

    /// Is size() == 0?.
    inline constexpr bool empty() const noexcept { return size() == 0; }

    // Element access

    /// Index operator.
    inline T& operator[](size_t n) const {
        check_invalidated();
        DCHECK_MSG(n < size(), "Trying to index at position "
                                   << n << " for span of size " << size());
        return *(data() + n);
    }

    /// Method with the same semantic as the index operator.
    inline T& at(size_t n) const {
        check_invalidated();
        DCHECK_MSG(n < size(), "Trying to index at position "
                                   << n << " for span of size " << size());
        return *(data() + n);
    }

    /// The first element of the slice.
    inline T& front() const {
        check_invalidated();
        DCHECK_MSG(size() != 0, "Call of front() with size() == 0");
        return *data();
    }

    /// The last element of the slice.
    inline T& back() const {
        check_invalidated();
        DCHECK_MSG(size() != 0, "Call of back() with size() == 0");
        return *(data() + size() - 1);
    }

    /// Pointer to the beginning of the slice.
    inline T* data() const {
        check_invalidated();
        return m_ptr;
    }

    /// Convert to a read-only span.
    inline constexpr operator span<T const>() const {
        check_invalidated();
        return span<T const>(data(), size());
    }

    /// Create a sub-slice from position `from` to position `to`.
    ///
    /// Leaving off the last parameter creates a suffix-slice.
    inline span<T> slice(size_t from = 0, size_t to = npos) const {
        check_invalidated();
        if (to == npos) {
            to = size();
        }
        DCHECK_MSG(0 <= from && from <= to && to <= size(),
                   "Slice with out-of-bound values "
                       << from << ".." << to << " for span of size " << size());
        auto r = span<T>(data() + from, to - from);
        IF_DEBUG(r.register_alive_check(m_alive_check));
        return r;
    }

    /// Copy the contents of another container into this one
    template<typename U>
    inline void copy_from(U&& other) {
        DCHECK_EQ(size(), other.size());
        for (size_t i = 0; i < size(); i++) {
            (*this)[i] = other[i];
        }
    }

    template<typename FmtFunction>
    inline std::ostream& debug_write(std::ostream& out, FmtFunction func) const {
        out << "[";
        bool first = true;
        for (auto const& e : *this) {
            if (first) {
                first = false;
            } else {
                out << ", ";
            }
            func(e, out);
        }
        out << "]";
        return out;
    }
    inline std::ostream& debug_write(std::ostream& out) const {
        return debug_write(out, write_map_char_to_int<std::remove_cv_t<T>>);
    }
    inline std::ostream& escaped_write(std::ostream& out) const {
        for (uint8_t byte : *this) {
            if (byte >= 32 && byte <= 126) {
                out << byte;
            } else {
                out << "\\x" << std::hex << uint32_t(byte) << std::dec;
            }
        }
        return out;
    }
};

template <typename T>
inline SB_FORCE_INLINE bool operator==(span<T> const& lhs, span<T> const& rhs) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <typename T>
inline SB_FORCE_INLINE bool operator!=(span<T> const& lhs, span<T> const& rhs) {
    return !(lhs == rhs);
}

template <typename T>
inline SB_FORCE_INLINE bool operator<(span<T> const& lhs, span<T> const& rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                        rhs.end());
}

template <typename T>
inline SB_FORCE_INLINE bool operator>(span<T> const& lhs, span<T> const& rhs) {
    return rhs < lhs;
}

template <typename T>
inline SB_FORCE_INLINE bool operator<=(span<T> const& lhs, span<T> const& rhs) {
    return !(lhs > rhs);
}

template <typename T>
inline SB_FORCE_INLINE bool operator>=(span<T> const& lhs, span<T> const& rhs) {
    return !(lhs < rhs);
}

} // namespace sacabench::util

/// Custom `std::ostream` operator for a `span<T>`
template <typename T>
inline std::ostream& operator<<(std::ostream& out,
                                sacabench::util::span<T> const& span) {
    return span.debug_write(out);
}
