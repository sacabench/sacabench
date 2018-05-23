/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/string.hpp>
#include <util/uint_types.hpp>

namespace sacabench::deep_shallow {

using u_char = util::character;

/// \brief Data on a single bucket. Contains the starting position in the suffix
///        array and if it is sorted.
template <typename sa_index_type>
struct bucket_information {
    sa_index_type starting_position;
    bool is_sorted;
};

static_assert(sizeof(bucket_information<util::uint40>) <= sizeof(uint64_t));

/// \brief A class which contains for every character combination the bucket
///        start and end positions.
template <typename sa_index_type>
struct bucket_data_container {
private:
    // This container has length `alphabet_size`^2.
    // It contains for every `alpha` a continous sequence of entries of bucket
    // information.
    util::container<bucket_information<sa_index_type>> bounds;
    size_t alphabet_size;

public:
    inline bucket_data_container() : bucket_data_container(0) {}

    inline bucket_data_container(const size_t _alphabet_size)
        : alphabet_size(_alphabet_size + 1) {
        const auto n = alphabet_size * alphabet_size;

        // Check if `n` overflowed
        DCHECK_GE(n, alphabet_size);

        bounds = util::make_container<bucket_information<sa_index_type>>(n);
    }

    inline void check_bounds(const u_char a, const u_char b) const {
        DCHECK_LT(a, alphabet_size);
        DCHECK_LT(b, alphabet_size);
    }

    inline void set_bucket_start(const u_char a,
                                 const size_t starting_position) const {
        bounds[a * alphabet_size].starting_position = starting_position;
    }

    /// \brief Assumes, that starting position is relative to the starting
    ///        position of the super-bucket.
    inline void set_subbucket_start(const u_char a, const u_char b,
                                    const size_t starting_position) const {
        bounds[a * alphabet_size + b].starting_position =
            starting_position + bounds[a * alphabet_size];
    }

    inline bool is_bucket_sorted(const u_char a, const u_char b) const {
        check_bounds(a, b);
        return bounds[a * alphabet_size + b].is_sorted;
    }

    inline void mark_bucket_sorted(const u_char a, const u_char b) {
        check_bounds(a, b);
        bounds[a * alphabet_size + b].is_sorted = true;
    }

    inline sa_index_type start_of_bucket(const u_char a,
                                         const u_char b) const {
        check_bounds(a, b);
        return bounds[a * alphabet_size + b].starting_position;
    }

    inline sa_index_type end_of_bucket(const u_char a,
                                       const u_char b) const {
        check_bounds(a, b);
        return bounds[a * alphabet_size + b + 1].starting_position;
    }

    inline sa_index_type size_of_bucket(const u_char a,
                                        const u_char b) const {
        check_bounds(a, b);
        return end_of_bucket(a, b) - start_of_bucket(a, b);
    }
};
} // namespace sacabench::deep_shallow
