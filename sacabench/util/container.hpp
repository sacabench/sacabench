/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "span.hpp"
#include <vector>

namespace sacabench::util {

template <typename element_type>
class custom_container {
private:
    std::vector<element_type> m_allocation;

public:
    /// Special value that means "the end of the slice".
    static constexpr size_t npos = -1ll;

    /// Create a container of size 0.
    inline custom_container() : m_allocation() {}

    /// Create a container with a given length.
    ///
    /// Each element is default-constructed.
    inline explicit custom_container(size_t size) {
        m_allocation.reserve(size);
        m_allocation.resize(size);
    }

    /// Create a container from a initializer list.
    inline custom_container(std::initializer_list<element_type> init)
        : custom_container(init.size()) {
        std::copy(init.begin(), init.end(), begin());
    }

    // Capacity

    /// Pointer to the beginning of the container.
    inline element_type* data() noexcept { return m_allocation.data(); }
    /// Const pointer to the beginning of the container.
    inline element_type const* data() const noexcept {
        return m_allocation.data();
    }

    /// Size of the container, in elements.
    inline size_t size() const noexcept { return m_allocation.size(); }

    /// Convert to a span.
    inline operator span<element_type>() {
        return span<element_type>(data(), size());
    }

    /// Convert to a const span.
    inline operator span<element_type const>() const {
        return span<element_type const>(data(), size());
    }

    ////////////////////////////////////////////////////////////////////////
    // The remaining methods below just delegate to the corresponding impls on
    // span<element_type>
    ////////////////////////////////////////////////////////////////////////

private:
    // Two internal helper methods that make delegating to span easier
    inline span<element_type> as_span() {
        return operator span<element_type>();
    }
    inline span<element_type const> as_span() const {
        return operator span<element_type const>();
    }

public:
    /// Is size() == 0?.
    inline bool empty() const noexcept { return as_span().empty(); }

    // Iterators

    /// Iterator to the begin of the container.
    inline element_type* begin() noexcept { return as_span().begin(); }
    /// Const iterator to the begin of the container.
    inline element_type const* begin() const noexcept {
        return as_span().begin();
    }

    /// Iterator to the end of the container.
    inline element_type* end() noexcept { return as_span().end(); }
    /// Const iterator to the end of the container.
    inline element_type const* end() const noexcept { return as_span().end(); }

    // Element access

    /// Index operator.
    inline element_type& operator[](size_t n) { return as_span()[n]; }
    /// Const index operator.
    inline element_type const& operator[](size_t n) const {
        return as_span()[n];
    }

    /// Method with the same semantic as the index operator.
    inline element_type& at(size_t n) { return as_span().at(n); }
    /// Const method with the same semantic as the index operator.
    inline element_type const& at(size_t n) const { return as_span().at(n); }

    /// The first element of the container.
    inline element_type& front() { return as_span().front(); }
    /// The first element of the container as const.
    inline element_type const& front() const { return as_span().front(); }

    /// The last element of the container.
    inline element_type& back() { return as_span().back(); }
    /// The last element of the container as const.
    inline element_type const& back() const { return as_span().back(); }

    /// Create a sub-slice from position `from` to position `to`.
    ///
    /// Leaving off the last parameter creates a suffix-slice.
    inline span<element_type> slice(size_t from = 0, size_t to = npos) {
        return as_span().slice(from, to);
    }

    /// Create a const sub-slice from position `from` to position `to`.
    ///
    /// Leaving off the last parameter creates a suffix-slice.
    inline span<element_type const> slice(size_t from = 0,
                                          size_t to = npos) const {
        return as_span().slice(from, to);
    }

    inline friend bool operator==(custom_container<element_type> const& lhs,
                                  custom_container<element_type> const& rhs) {
        return lhs.as_span() == rhs.as_span();
    }

    inline friend bool operator!=(custom_container<element_type> const& lhs,
                                  custom_container<element_type> const& rhs) {
        return lhs.as_span() != rhs.as_span();
    }

    inline friend bool operator<(custom_container<element_type> const& lhs,
                                 custom_container<element_type> const& rhs) {
        return lhs.as_span() < rhs.as_span();
    }

    inline friend bool operator>(custom_container<element_type> const& lhs,
                                 custom_container<element_type> const& rhs) {
        return lhs.as_span() > rhs.as_span();
    }

    inline friend bool operator<=(custom_container<element_type> const& lhs,
                                  custom_container<element_type> const& rhs) {
        return lhs.as_span() <= rhs.as_span();
    }

    inline friend bool operator>=(custom_container<element_type> const& lhs,
                                  custom_container<element_type> const& rhs) {
        return lhs.as_span() >= rhs.as_span();
    }
};

template <typename element_type>
using container = custom_container<element_type>;

/**\brief Creates a container with space for exactly for `size` elements.
 */
template <typename element_type>
container<element_type> make_container(size_t size) {
    return container<element_type>(size);
}

/**\brief Creates a container as a copy of the elements of a `span`.
 */
template <typename element_type>
container<element_type> make_container(span<element_type> s) {
    container<element_type> r = make_container<element_type>(s.size());
    std::copy(s.begin(), s.end(), r.begin());
    return r;
}
} // namespace sacabench::util

/// Custom `std::ostream` operator for a `span<T>`
template <typename T>
inline std::ostream&
operator<<(std::ostream& out, sacabench::util::container<T> const& container) {
    return out << container.slice();
}

/******************************************************************************/
