/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow {

using u_char = util::character;

/// \brief Data on a single bucket. Contains the starting position in the suffix
///        array and if it is sorted.
template <typename sa_index_type>
struct bucket_information {
    sa_index_type starting_position;
    bool is_sorted;
};

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

    inline bucket_data_container(size_t _alphabet_size)
        : alphabet_size(_alphabet_size) {
        const auto n = _alphabet_size * _alphabet_size;

        // Check if `n` overflowed.
        DCHECK_GE(n, _alphabet_size);

        bounds = util::make_container<bucket_information<sa_index_type>>(n);
    }

    inline sa_index_type start_of_bucket(u_char alpha, u_char beta) {
        DCHECK_LT(alpha, alphabet_size);
        DCHECK_LT(beta, alphabet_size);
        return bounds[alpha * alphabet_size + beta].starting_position;
    }

    inline sa_index_type end_of_bucket(u_char alpha, u_char beta) {
        DCHECK_LT(alpha, alphabet_size);
        DCHECK_LT(beta, alphabet_size);
        return bounds[alpha * alphabet_size + beta + 1].starting_position;
    }

    inline sa_index_type size_of_bucket(u_char alpha, u_char beta) {
        DCHECK_LT(alpha, alphabet_size);
        DCHECK_LT(beta, alphabet_size);
        return end_of_bucket(alpha, beta) - start_of_bucket(alpha, beta);
    }

    inline bool is_bucket_sorted(u_char alpha, u_char beta) {
        DCHECK_LT(alpha, alphabet_size);
        DCHECK_LT(beta, alphabet_size);
        return bounds[alpha * alphabet_size + beta].is_sorted;
    }
};
} // namespace sacabench::deep_shallow
