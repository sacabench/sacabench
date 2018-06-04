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
    util::container<uint16_t> anchor;

    /// \brief Contains for the index in `anchor` the position of the suffix in
    ///        the suffix_array.
    util::container<sa_index_type> offset;

public:
    inline anchor_data(const size_t text_length) {
        const double n_segments =
            text_length / static_cast<double>(SEGMENT_LENGTH);
        const size_t int_n_segments = static_cast<size_t>(n_segments) + 1;

        std::cout << "Anzahl Segmente: " << n_segments << ", " << int_n_segments
                  << std::endl;

        anchor = util::make_container<uint16_t>(int_n_segments);
        offset = util::make_container<sa_index_type>(int_n_segments);

        // Initialize every `anchor` with the null value.
        for (size_t i = 0; i < anchor.size(); ++i) {
            anchor[i] = NULLBYTES;
        }
    }

    inline void update_anchor(sa_index_type n_anchor, sa_index_type n_offset) {
        std::cout << "Updating " << n_anchor << " to point to " << n_offset
                  << std::endl;

        // This is the segment the n_anchor belongs to.
        size_t segment_idx = n_anchor / SEGMENT_LENGTH;

        // This is the relative index of n_anchor in its segment.
        uint16_t relative_anchor = n_anchor % SEGMENT_LENGTH;

        // This is the currently saved anchor for n_anchor's segment.
        uint16_t anchor_entry = anchor[segment_idx];

        // If there is no entry yet, or the new anchor is nearer to the left
        // edge of the segment, update it.
        if (anchor_entry == NULLBYTES || relative_anchor < anchor_entry) {
            anchor[segment_idx] = relative_anchor;
            offset[segment_idx] = n_offset;
        }
    }

    inline std::optional<sa_index_type> get_offset(sa_index_type n_anchor) {
        // This is the segment the n_anchor belongs to.
        size_t segment_idx = n_anchor / SEGMENT_LENGTH;

        // This is the currently saved anchor for n_anchor's segment.
        uint16_t anchor_entry = anchor[segment_idx];

        // Check, if value has been initialized and return it, or None.
        if (anchor_entry == NULLBYTES) {
            return std::nullopt;
        } else {
            return offset[segment_idx];
        }
    }
};
} // namespace sacabench::deep_shallow
