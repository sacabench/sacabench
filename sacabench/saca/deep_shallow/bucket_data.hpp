/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/string.hpp>
#include <util/uint_types.hpp>
#include <util/kd_array.hpp>

namespace sacabench::deep_shallow {

template <typename T>
using span = util::span<T>;
using u_char = util::character;

/// \brief Data on a single bucket. Contains the starting position in the suffix
///        array and if it is sorted.
template <typename sa_index_type>
struct bucket_information {
    sa_index_type starting_position;
    bool is_sorted;
};

// Assert, that the free bits in a 64 bit type, that are not used by the
// suffix index type (32, 40, 48 bit) are used to store the boolean `is_sorted`.
static_assert(sizeof(bucket_information<uint32_t>) <= sizeof(uint64_t));
static_assert(sizeof(bucket_information<util::uint40>) <= sizeof(uint64_t));
static_assert(sizeof(bucket_information<util::uint48>) <= sizeof(uint64_t));

/// \brief A class which contains for every character combination the bucket
///        start and end positions.
template <typename sa_index_type>
struct bucket_data_container {
private:
    // This is the real alphabet size, containing the SENTINEL symbol.
    size_t real_alphabet_size;

    // This container has length `alphabet_size`^2.
    // It contains for every `alpha` a continous sequence of entries of bucket
    // information.
    util::kd_array<bucket_information<sa_index_type>, 2> bounds;
    sa_index_type end_of_last_bucket = 0;

public:
    inline bucket_data_container() : bucket_data_container(0) {}

    inline bucket_data_container(const size_t alphabet_size)
        : real_alphabet_size(alphabet_size + 1), bounds({real_alphabet_size,real_alphabet_size}) {}

    inline void check_bounds(const u_char a, const u_char b) const {
        DCHECK_LT(a, real_alphabet_size);
        DCHECK_LT(b, real_alphabet_size);

        // "Use" `a` and `b` so that the compiler doesn't warn about them
        // being unused.
        (void)a;
        (void)b;
    }

    inline void set_bucket_bounds(
        const util::container<util::sort::bucket>& bucket_bounds) {
        DCHECK_EQ(bucket_bounds.size(), bounds.size()[0] * bounds.size()[1]);

        for (size_t i = 0; i < bucket_bounds.size(); ++i) {
            const util::character alpha = i / real_alphabet_size;
            const util::character beta = i % real_alphabet_size;

            // std::cout << (size_t)alpha << ", " << (size_t)beta << std::endl;

            bounds[{alpha,beta}].starting_position = bucket_bounds[i].position;
        }

        end_of_last_bucket = bucket_bounds[bucket_bounds.size() - 1].position +
                             bucket_bounds[bucket_bounds.size() - 1].count;
    }

    inline bool is_bucket_sorted(const u_char a, const u_char b) const {
        check_bounds(a, b);
        return bounds[{a,b}].is_sorted;
    }

    inline void mark_bucket_sorted(const u_char a, const u_char b) {
        check_bounds(a, b);
        bounds[{a,b}].is_sorted = true;
    }

    inline sa_index_type start_of_bucket(const u_char a, const u_char b) const {
        check_bounds(a, b);
        return bounds[{a,b}].starting_position;
    }

    inline sa_index_type end_of_bucket(const u_char a, const u_char b) const {
        check_bounds(a, b);
        if (a == b && b == real_alphabet_size - 1) {
            return end_of_last_bucket;
        } else {
            return bounds[{a,b+1u}].starting_position;
        }
    }

    inline sa_index_type size_of_bucket(const u_char a, const u_char b) const {
        check_bounds(a, b);
        return end_of_bucket(a, b) - start_of_bucket(a, b);
    }
};
} // namespace sacabench::deep_shallow
