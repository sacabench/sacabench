/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>

#include <util/container.hpp>

#include "parameters.hpp"

namespace sacabench::deep_shallow {

template <typename sa_index_type>
class anchor_data {
    /// \brief This value represents "no suffix from this segment has been
    ///        sorted yet".
    static constexpr uint16_t NULLBYTES = -1;

    /// \brief Contains for every segment either NULL or the suffix index of the
    ///        leftmost suffix belonging to an already sorted bucket. This
    ///        "offset" is relative to the segment start, and under the
    ///        assumption that SEGMENT_LENGTH < 2^16, we can store it in a
    ///        16-bit type.
    util::container<uint16_t> relative_positions_in_segments;

    /// \brief Contains for the index in `relative_positions_in_segments`
    ///        ("offset") the position of the suffix in its already-sorted
    ///        bucket (in their paper, this array is called "Anchor".). Because
    ///        this is an index into the suffix array, it cannot be smaller
    ///        than `sa_index_type`.
    util::container<sa_index_type> position_in_suffixarray;

public:
    inline anchor_data(const size_t text_length) {
        // This is the number of segments we store. We use integer division
        // plus one, because we actually need to round up.
        const size_t int_n_segments = (text_length / SEGMENT_LENGTH) + 1;

        relative_positions_in_segments =
            util::make_container<uint16_t>(int_n_segments);
        position_in_suffixarray =
            util::make_container<sa_index_type>(int_n_segments);

        // Initialize every `anchor` with the null value.
        for (size_t i = 0; i < relative_positions_in_segments.size(); ++i) {
            relative_positions_in_segments[i] = NULLBYTES;
        }
    }

    inline void update_anchor(sa_index_type suffix,
                              sa_index_type position_in_sa) {
        // This is the segment the suffix belongs to.
        const size_t segment_idx = suffix / SEGMENT_LENGTH;

        // This is the relative index of suffix in its segment.
        const uint16_t relative = suffix % SEGMENT_LENGTH;

        // This is the currently saved anchor for n_anchor's segment.
        const uint16_t leftmost_suffix =
            relative_positions_in_segments[segment_idx];

        // If there is no entry yet, or the new anchor is nearer to the left
        // edge of the segment, update it.
        if (leftmost_suffix == NULLBYTES || relative < leftmost_suffix) {
            relative_positions_in_segments[segment_idx] = relative;
            position_in_suffixarray[segment_idx] = position_in_sa;
        }
    }

    inline std::optional<uint16_t>
    get_leftmost_relative_position(const sa_index_type suffix) const {
        // This is the segment the suffix belongs to.
        const size_t segment_idx = suffix / SEGMENT_LENGTH;

        // This is the currently saved anchor for suffix's segment.
        const uint16_t leftmost_relative_position =
            relative_positions_in_segments[segment_idx];

        // Check, if value has been initialized and return it, or None.
        if (leftmost_relative_position == NULLBYTES) {
            return std::nullopt;
        } else {
            return leftmost_relative_position;
        }
    }

    inline std::optional<sa_index_type>
    get_leftmost_position(const sa_index_type suffix) const {
        // This is the segment the suffix belongs to.
        const size_t segment_idx = suffix / SEGMENT_LENGTH;

        // This is the currently saved anchor for suffix's segment.
        const uint16_t leftmost_relative_position =
            relative_positions_in_segments[segment_idx];

        // Check, if value has been initialized and return it, or None.
        if (leftmost_relative_position == NULLBYTES) {
            return std::nullopt;
        } else {
            return leftmost_relative_position + segment_idx * SEGMENT_LENGTH;
        }
    }

    inline sa_index_type
    get_position_in_suffixarray(const sa_index_type n_anchor) const {
        // This is the segment the suffix belongs to.
        size_t segment_idx = n_anchor / SEGMENT_LENGTH;

        // This is the currently saved offset for suffix's segment.
        return position_in_suffixarray[segment_idx];
    }
};
} // namespace sacabench::deep_shallow
