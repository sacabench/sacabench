/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>

namespace sacabench::util {

// Forward-declare helper classes.
template <typename content, size_t k>
class incomplete_kd_array_access;
template <typename content, size_t k>
class incomplete_kd_array_access_const;

/// \brief A class which holds a large memory chunk and supports
///        multi-dimensional indexing into it. You can use either
///         array.get({a,b,c})/array.get_mut({a,b,c}), array[a][b][c] or
///         array[{a,b,c}] to access the array elements.
template <typename content, size_t k>
class kd_array {
private:
    /// \brief This length `k` vector contains the size of this array in every
    ///        dimension.
    std::array<size_t, k> sizes;

    /// \brief The big chunk of memory this array allocated and manages.
    container<content> memory;

public:
    /// \brief Creates a new array with the given sizes for every dimension.
    inline kd_array(std::array<size_t, k>&& _sizes) : sizes(_sizes) {
        // Calculate needed size for the memory block.
        size_t size_in_objects = 1;
        for (size_t dim_n : sizes) {
            size_in_objects *= dim_n;
        }

        // Allocate Memory
        memory = make_container<content>(size_in_objects);
    }

    /// \brief Calculates the actual index into the memory chunk for the given
    ///        kd position.
    inline size_t index(const std::array<size_t, k>& idx) const {
        size_t offset = 1;
        size_t final_idx = 0;

        // Because we can't loop until i = 0, we loop until i = 1 and subtract
        // one.
        for (size_t i = k; i > 0; --i) {
            size_t j = i - 1;
            final_idx += idx[j] * offset;
            offset *= sizes[j];
        }

        return final_idx;
    }

    /// \brief Returns the element at the given kd position.
    inline const content& get(const std::array<size_t, k>& idx) const {
        return memory[index(idx)];
    }

    /// \brief Returns the element at the given kd position.
    inline content& get_mut(const std::array<size_t, k>& idx) {
        return memory[index(idx)];
    }

    /// \brief Returns the element at the given kd position, using the
    ///        array[{1,2,3}] syntax.
    inline content& operator[](const std::array<size_t, k>& idx) {
        return get_mut(idx);
    }

    /// \brief Returns the element at the given kd position, using the
    ///        array[{1,2,3}] syntax. Const variant.
    inline const content& operator[](const std::array<size_t, k>& idx) const {
        return get(idx);
    }

    /// \brief Updates the element at the given kd position.
    inline void set(const std::array<size_t, k>& idx, const content& v) {
        memory[index(idx)] = v;
    }

    /// \brief Updates the element at the given kd position.
    inline void set(const std::array<size_t, k>& idx, content&& v) {
        memory[index(idx)] = std::move(v);
    }

    /// \brief A
    inline std::array<size_t, k> size() const {
        return sizes;
    }

    /// \brief Constructs a incomplete_kd_array_access objects and starts
    ///        indexing into the memory.
    inline incomplete_kd_array_access<content, k> operator[](const size_t idx) {
        return incomplete_kd_array_access<content, k>(*this, idx);
    }

    /// \brief Constructs a incomplete_kd_array_access_const objects and starts
    ///        indexing into the memory. This function yields a read-only
    ///        reference.
    inline incomplete_kd_array_access_const<content, k>
    operator[](const size_t idx) const {
        return incomplete_kd_array_access_const<content, k>(*this, idx);
    }
};

/// \brief A helper class to support [a][b][c]... syntax. It support indexing
///        (with []) and implicit casting into the target type.
template <typename content, size_t k>
class incomplete_kd_array_access {
private:
    kd_array<content, k>& array;
    std::array<size_t, k> indices;
    size_t n_indices;

public:
    inline incomplete_kd_array_access(kd_array<content, k>& _array,
                                      const size_t index)
        : array(_array), indices(), n_indices(1) {
        indices[0] = index;
    }

    inline incomplete_kd_array_access(incomplete_kd_array_access&& other,
                                      const size_t index)
        : array(other.array), indices(other.indices),
          n_indices(other.n_indices + 1) {
        indices[n_indices - 1] = index;
    }

    inline incomplete_kd_array_access<content, k>
    operator[](const size_t& index) {
        return incomplete_kd_array_access(std::move(*this), index);
    }

    inline operator content&() const {
        DCHECK_MSG(n_indices == k, "incomplete kd-array access");
        return array.get_mut(indices);
    }

    inline const content& operator=(const content& other) {
        DCHECK_MSG(n_indices == k, "incomplete kd-array access");
        array.set(indices, other);
        return other;
    }

    inline void operator=(content&& other) {
        DCHECK_MSG(n_indices == k, "incomplete kd-array access");
        array.set(indices, std::move(other));
    }
};

/// \brief A helper class to support [a][b][c]... syntax. It support indexing
///        (with []) and implicit casting into the target type. This variant
///        is used, if the kd-array is read only ("const").
template <typename content, size_t k>
class incomplete_kd_array_access_const {
private:
    const kd_array<content, k>& array;
    std::array<size_t, k> indices;
    size_t n_indices;

public:
    inline incomplete_kd_array_access_const(const kd_array<content, k>& _array,
                                            const size_t index)
        : array(_array), indices(), n_indices(1) {
        indices[0] = index;
    }

    inline incomplete_kd_array_access_const(
        const incomplete_kd_array_access_const&& other, const size_t index)
        : array(other.array), indices(other.indices),
          n_indices(other.n_indices + 1) {
        indices[n_indices - 1] = index;
    }

    inline incomplete_kd_array_access_const<content, k>
    operator[](const size_t index) const {
        return incomplete_kd_array_access_const(std::move(*this), index);
    }

    inline operator const content&() const {
        DCHECK_MSG(n_indices == k, "incomplete kd-array access");
        return array.get(indices);
    }

    // Delete assignment operators, because this is the const-variant.
    inline void operator=(const content) = delete;
};

/// \brief A helper definition for a 2d-array.
template <typename content>
using array2d = kd_array<content, 2>;

} // namespace sacabench::util
