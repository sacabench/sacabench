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
    ///        leftmost suffix belonging to an already sorted bucket. The anchor
    ///        is relative to the segment start.
    util::container<sa_index_type> position_in_suffixarray;

    /// \brief Contains for the index in `anchor` the position of the suffix in
    ///        the suffix_array.
    util::container<uint16_t> relative_positions_in_segments;

public:
    inline anchor_data(const size_t text_length) {
        const double n_segments =
            text_length / static_cast<double>(SEGMENT_LENGTH);
        const size_t int_n_segments = static_cast<size_t>(n_segments) + 1;

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
        // std::cout << "Updating " << n_anchor << " to point to " << n_offset
        //           << std::endl;

        // This is the segment the n_anchor belongs to.
        const size_t segment_idx = suffix / SEGMENT_LENGTH;

        // This is the relative index of n_anchor in its segment.
        const uint16_t relative_anchor = suffix % SEGMENT_LENGTH;

        // This is the currently saved anchor for n_anchor's segment.
        const uint16_t leftmost_suffix =
            relative_positions_in_segments[segment_idx];

        // If there is no entry yet, or the new anchor is nearer to the left
        // edge of the segment, update it.
        if (leftmost_suffix == NULLBYTES || relative_anchor < leftmost_suffix) {
            relative_positions_in_segments[segment_idx] = relative_anchor;
            position_in_suffixarray[segment_idx] = position_in_sa;
        }
    }

    inline std::optional<uint16_t>
    get_leftmost_relative_position(const sa_index_type suffix) const {
        // This is the segment the n_anchor belongs to.
        const size_t segment_idx = suffix / SEGMENT_LENGTH;

        // This is the currently saved anchor for n_anchor's segment.
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
        // This is the segment the n_anchor belongs to.
        const size_t segment_idx = suffix / SEGMENT_LENGTH;

        // This is the currently saved anchor for n_anchor's segment.
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
        // This is the segment the n_anchor belongs to.
        size_t segment_idx = n_anchor / SEGMENT_LENGTH;

        // This is the currently saved offset for n_anchor's segment.
        return position_in_suffixarray[segment_idx];
    }
};
} // namespace sacabench::deep_shallow
