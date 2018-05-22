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

template <typename sa_index_type>
struct bucket_bounds {
private:
    // This container has length `alphabet_size`^2.
    // It contains for every `alpha` a continous sequence of entries of bucket
    // starting positions.
    util::container<sa_index_type> bounds;
    size_t alphabet_size;

public:
    bucket_bounds() : bucket_bounds(0) {}

    bucket_bounds(size_t _alphabet_size) : alphabet_size(_alphabet_size) {
        bounds = util::make_container<sa_index_type>(_alphabet_size *
                                                     _alphabet_size);
    }

    sa_index_type start_of_bucket(u_char alpha, u_char beta) {
        return bounds[alpha * alphabet_size + beta];
    }

    sa_index_type end_of_bucket(u_char alpha, u_char beta) {
        return bounds[alpha * alphabet_size + beta + 1];
    }

    sa_index_type size_of_bucket(u_char alpha, u_char beta) {
        return end_of_bucket(alpha, beta) - start_of_bucket(alpha, beta);
    }
};
} // namespace sacabench::deep_shallow
