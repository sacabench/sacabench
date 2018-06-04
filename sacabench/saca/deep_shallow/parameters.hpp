/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

namespace sacabench::deep_shallow {

/// \brief Instead of continuing to sort with shallow_sort, we switch to
///        deep_sort as this depth.
constexpr auto DEEP_SORT_DEPTH = 50;

/// \brief We use blind sort on sets which are smaller than this threshold.
///        This has a direct effect on the memory footprint of the algorithm.
constexpr auto BLIND_SORT_THRESHOLD = 100;

/// \brief To speed up sorting, we store meta data for every one of the
///        `text_length/SEGMENT_LENGTH` segments. Each of the segments has
///        length `SEGMENT_LENGTH`.
constexpr auto SEGMENT_LENGTH = 200;
static_assert(SEGMENT_LENGTH < std::numeric_limits<uint16_t>::max());
} // namespace sacabench::deep_shallow
