/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <memory>
#include <sstream>
#include <vector>

#include "macros.hpp"
#include "span.hpp"
#include "stacktrace.hpp"

namespace sacabench::util {

template <typename element_type>
class container;

class allow_container_copy {
    inline static bool s_allow_copy = false;
    bool m_last_allow_copy;

    template <typename element_type>
    friend class container;

public:
    inline allow_container_copy() {
        m_last_allow_copy = s_allow_copy;
        s_allow_copy = true;
    }
    inline ~allow_container_copy() { s_allow_copy = m_last_allow_copy; }
    inline allow_container_copy(allow_container_copy const&) = delete;
    inline allow_container_copy(allow_container_copy&&) = delete;
};

template <typename element_type>
class container {
private:
    std::unique_ptr<element_type[]> m_allocation;
    size_t m_size = 0;

    inline static void copy_check(span<element_type const> other) {
        (void)other;
        IF_DEBUG({
            constexpr bool make_error = false;

            if (!allow_container_copy::s_allow_copy && !other.empty()) {
                std::stringstream ss;

                bool possible_trigger = false;
                bool triggered = false;
                backtrace::Backtrace(1, true, [&](auto r) {
                    if (possible_trigger) {
                        possible_trigger = false;

                        triggered =
                            r.function_namespace.starts_with("sacabench");
                        if (triggered) {
                            ss.str("");
                            ss << r.function_namespace
                               << "::" << r.function_name;
                        }
                    }
                    if (r.function_namespace.starts_with(
                            "sacabench::util::container") ||
                        r.function_namespace_and_name.starts_with(
                            "sacabench::util::make_container") ||
                        r.function_namespace_and_name.starts_with(
                            "sacabench::util::make_string")) {
                        possible_trigger = true;
                    }
                });

                if (triggered) {
                    if (make_error) {
                        DCHECK_MSG(
                            false,
                            "Copy of non-empty container or string in "
                            "function\n`"
                                << ss.str()
                                << "`"
                                   "\nIf the copy was intentional, "
                                   "consider using `.make_copy()` instead.");
                    } else {
                        std::cerr << "WARNING: Copy of non-empty container or "
                                     "string in function `"
                                  << ss.str()
                                  << "`"
                                     "\n";
                    }
                }
            }
        })
    }

public:
    /// Special value that means "the end of the slice".
    static constexpr size_t npos = -1ll;

    /// Create a container of size 0.
    inline container() {}

    /// Create a container with a given length.
    ///
    /// Each element is default-constructed.
    inline explicit container(size_t size) {
        if (size != 0ull) {
            DCHECK_MSG(size * sizeof(element_type) < (1ull << 48),
                       "Trying to create a `container` with a size of "
                           << size
                           << ", which is unrealistic large. This likely "
                              "happened through a underflow "
                              "when calculating a size `< 0`.")

            m_size = size;
            m_allocation = std::make_unique<element_type[]>(size);
        }
    }

    /// Create a container from a initializer list.
    inline container(std::initializer_list<element_type> init)
        : container(init.size()) {
        std::copy(init.begin(), init.end(), begin());
    }

    /// Create a container by copying from a span.
    inline container(span<element_type const> span) : container(span.size()) {
        copy_check(span);
        std::copy(span.begin(), span.end(), begin());
    }
    /// Create a container by copying from a span.
    inline container(span<element_type> span) : container(span.size()) {
        copy_check(span);
        std::copy(span.begin(), span.end(), begin());
    }

    inline container(container const& other) { *this = other.as_span(); }
    inline container& operator=(container const& other) {
        *this = other.as_span();
        return *this;
    }
    inline container(container&& other) = default;
    inline container& operator=(container&& other) = default;

    // Capacity

    /// Pointer to the beginning of the container.
    inline element_type* data() {
        DCHECK_EQ((m_size != 0), (m_allocation.get() != nullptr));
        return m_allocation.get();
    }
    /// Const pointer to the beginning of the container.
    inline element_type const* data() const {
        DCHECK_EQ((m_size != 0), (m_allocation.get() != nullptr));
        return m_allocation.get();
    }

    /// Size of the container, in elements.
    inline size_t size() const {
        DCHECK_EQ((m_size != 0), (m_allocation.get() != nullptr));
        return m_size;
    }

    /// Convert to a span.
    inline operator span<element_type>() {
        return span<element_type>(data(), size());
    }

    /// Convert to a const span.
    inline operator span<element_type const>() const {
        return span<element_type const>(data(), size());
    }

    /// Explicit copy operation.
    ///
    /// This exists instead of the implicit copy constructor
    /// to catch bugs, and make copies explicit.
    inline container make_copy() const {
        container r = container(size());
        std::copy(begin(), end(), r.begin());
        return r;
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

    // container OP container

    inline friend bool operator==(container<element_type> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() == rhs.slice();
    }

    inline friend bool operator!=(container<element_type> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() != rhs.slice();
    }

    inline friend bool operator<(container<element_type> const& lhs,
                                 container<element_type> const& rhs) {
        return lhs.slice() < rhs.slice();
    }

    inline friend bool operator>(container<element_type> const& lhs,
                                 container<element_type> const& rhs) {
        return lhs.slice() > rhs.slice();
    }

    inline friend bool operator<=(container<element_type> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() <= rhs.slice();
    }

    inline friend bool operator>=(container<element_type> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() >= rhs.slice();
    }

    // span OP container

    inline friend bool operator==(span<element_type const> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() == rhs.slice();
    }

    inline friend bool operator!=(span<element_type const> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() != rhs.slice();
    }

    inline friend bool operator<(span<element_type const> const& lhs,
                                 container<element_type> const& rhs) {
        return lhs.slice() < rhs.slice();
    }

    inline friend bool operator>(span<element_type const> const& lhs,
                                 container<element_type> const& rhs) {
        return lhs.slice() > rhs.slice();
    }

    inline friend bool operator<=(span<element_type const> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() <= rhs.slice();
    }

    inline friend bool operator>=(span<element_type const> const& lhs,
                                  container<element_type> const& rhs) {
        return lhs.slice() >= rhs.slice();
    }

    // container OP span

    inline friend bool operator==(container<element_type> const& lhs,
                                  span<element_type const> const& rhs) {
        return lhs.slice() == rhs.slice();
    }

    inline friend bool operator!=(container<element_type> const& lhs,
                                  span<element_type const> const& rhs) {
        return lhs.slice() != rhs.slice();
    }

    inline friend bool operator<(container<element_type> const& lhs,
                                 span<element_type const> const& rhs) {
        return lhs.slice() < rhs.slice();
    }

    inline friend bool operator>(container<element_type> const& lhs,
                                 span<element_type const> const& rhs) {
        return lhs.slice() > rhs.slice();
    }

    inline friend bool operator<=(container<element_type> const& lhs,
                                  span<element_type const> const& rhs) {
        return lhs.slice() <= rhs.slice();
    }

    inline friend bool operator>=(container<element_type> const& lhs,
                                  span<element_type const> const& rhs) {
        return lhs.slice() >= rhs.slice();
    }
};

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
    return container<element_type>(s);
}
} // namespace sacabench::util

/// Custom `std::ostream` operator for a `span<T>`
template <typename T>
inline std::ostream&
operator<<(std::ostream& out, sacabench::util::container<T> const& container) {
    return out << container.slice();
}

/******************************************************************************/
