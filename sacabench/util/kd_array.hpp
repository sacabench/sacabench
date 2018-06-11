#pragma once

#include <util/container.hpp>

namespace sacabench::util {

/// \brief A class which holds a large memory chunk and supports
///        multi-dimensional indexing into it.
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

        // std::cout << "Allocating space for " << size_in_objects << " objects."
        //           << std::endl;

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
    inline content get(const std::array<size_t, k>& idx) const {
        return memory[index(idx)];
    }

    /// \brief Updates the element at the given kd position.
    inline void set(const std::array<size_t, k>& idx, content v) {
        memory[index(idx)] = v;
    }
};

/// \brief A helper definition for a 2d-array.
template <typename content>
using array2d = kd_array<content, 2>;

} // namespace sacabench::util
