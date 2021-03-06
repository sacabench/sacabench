#pragma once

#include "utils.hpp"
#include <iostream>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::div_suf_sort {

template <typename sa_index>
inline static void
induce_s_suffixes(util::string_span input, buckets<sa_index>& buckets,
                  util::span<sa_index> sa, const size_t max_character) {
    // bit mask: 1000...000
    constexpr sa_index NEGATIVE_MASK = size_t(1) << (sizeof(sa_index) * 8 - 1);

    for (size_t c0 = max_character; c0 > '\1'; --c0) {
        // c1 = c0 - 1
        // start at rightmost position of L-bucket of c1
        size_t interval_start = buckets.l_buckets[c0] - sa_index(1);
        // end at RMS-bucket[c1, c1 + 1]
        size_t interval_end =
            buckets.s_buckets[buckets.get_rms_bucket_index(c0 - 1, c0)];
        if (interval_end == 0) {
            break;
        }
        // induce positions for each suffix in range
        // +1 to allow i reaching 0 (because of unsigned types)
        for (size_t i = interval_start; i >= interval_end; --i) {
            // Index 0 found - cannot induce anything -> skip
            if (sa[i] == sa_index(0)) {
                continue;
            }

            if ((sa[i] & NEGATIVE_MASK) == 0) {
                // entry is not negative -> induce predecessor
                // insert suffix i-1 at rightmost free index of
                // associated S-bucket
                size_t destination_bucket = buckets.get_s_bucket_index(
                    input[size_t(sa[i]) - 1], input[sa[i]]);
                // This check for l-type is sufficient, because we know that
                // it's successor is an s-type (i.e. different, greater symbol
                // needed)
                if (size_t(sa[i]) - 1 > 0 &&
                    input[size_t(sa[i])-2] > input[size_t(sa[i])-1]) {
                    //sa_types::is_l_type(size_t(sa[i]) - 2, suffix_types)) {

                    // Check if index is used to induce in current step
                    // (induce s-suffixes)
                    // Prefix/Postfix-operators not supported for uint40
                    sa[buckets.s_buckets[destination_bucket]] =
                        (size_t(sa[i]) - 1) ^ NEGATIVE_MASK;
                    buckets.s_buckets[destination_bucket] =
                        buckets.s_buckets[destination_bucket] - sa_index(1);
                } else {
                    // Prefix/Postfix-operators not supported for uint40
                    sa[buckets.s_buckets[destination_bucket]] =
                        sa[i] - sa_index(1);
                    buckets.s_buckets[destination_bucket] =
                        buckets.s_buckets[destination_bucket] - sa_index(1);
                }
            }
            // toggle flag
            sa[i] = sa[i] ^ NEGATIVE_MASK;
        }
    }

    // "$" is the first index
    sa[0] = input.size() - 1;

    // if predecessor is S-suffix
    if (input[input.size() - 2] < input[input.size() - 1]) {
        sa[0] = sa[0] | NEGATIVE_MASK;
    }
}

template <typename sa_index>
inline static void induce_l_suffixes(util::string_span input,
                                     buckets<sa_index>& buckets,
                                     util::span<sa_index> sa) {
    // bit mask: 1000...000
    constexpr sa_index NEGATIVE_MASK = size_t(1) << (sizeof(sa_index) * 8 - 1);
    size_t insert_position;
    for (size_t i = 0; i < sa.size(); ++i) {
        // Index 0 has no predecessor -> skip
        if (sa[i] == sa_index(0)) {
            continue;
        } else if ((sa[i] & NEGATIVE_MASK) > 0) {
            // entry is negative: sa[i]-1 already induced -> remove flag
            sa[i] = sa[i] ^ NEGATIVE_MASK;
        } else {
            // predecessor has yet to be induced
            insert_position = buckets.l_buckets[input[size_t(sa[i]) - 1]];
            buckets.l_buckets[input[size_t(sa[i]) - 1]] =
                buckets.l_buckets[input[size_t(sa[i]) - 1]] + sa_index(1);
            DCHECK_LT(insert_position, input.size());
            sa[insert_position] = sa[i] - sa_index(1);
            if (size_t(sa[i]) - 1 > 0 &&
                input[size_t(sa[i]) - 2] < input[size_t(sa[i]) - 1]) {
                // predecessor of induced index is S-suffix
                sa[insert_position] = sa[insert_position] | NEGATIVE_MASK;
            }
        }
    }
}
} // namespace sacabench::div_suf_sort
