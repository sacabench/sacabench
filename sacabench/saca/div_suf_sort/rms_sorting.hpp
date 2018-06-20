#pragma once

#include "comparison_fct.hpp"
#include "utils.hpp"
#include <util/assertions.hpp>
#include <util/bits.hpp>
#include <util/container.hpp>
#include <util/sort/introsort.hpp>
#include <util/span.hpp>

namespace sacabench::div_suf_sort {

template <typename sa_index>
inline static void set_unsorted_rms_substring_intervals(
    rms_suffixes<sa_index>& rms_suf, compare_rms_substrings<sa_index> cmp,
    sa_index interval_start, sa_index interval_end) {
    sa_index elem, compare_to;
    // Last element to be compared to its predecessor: interval_start + 1
    // interval_end must not contain last element of interval
    // bool less, greater;
    for (sa_index pos = interval_end - 1; interval_start < pos; --pos) {
        elem = rms_suf.relative_indices[pos - 1];
        compare_to = rms_suf.relative_indices[pos];
        // None of the substrings is smaller than the other (i.e. the same)
        /*less = cmp(elem, compare_to);
        greater = cmp(compare_to, elem);
        std::cout << "Index " << elem << " is smaller than " << compare_to
                  << ":" << less << std::endl;
        std::cout << "Index " << elem << " is greater than " << compare_to
                  << ":" << greater << std::endl;*/

        if (!(cmp(elem, compare_to) || cmp(compare_to, elem))) {
            rms_suf.relative_indices[pos] |= utils<sa_index>::NEGATIVE_MASK;
            /*std::cout << "Negated index " << compare_to
                      << " because it's the same substring as its predecessor."
                      << std::endl;*/
        }
    }
}

template <typename sa_index>
inline static void sort_rms_substrings(rms_suffixes<sa_index>& rms_suf,
                                       const size_t max_character_value,
                                       buckets<sa_index>& sa_buckets) {
    std::cout << "rel_ind: ";
    for(sa_index pos = 0; pos < rms_suf.relative_indices.size(); ++pos) {
        std::cout << rms_suf.relative_indices[pos] << " ";
    }
    std::cout << std::endl;


    // Compute RMS-Substrings (tupel with start-/end-position)
    auto substrings_container = extract_rms_substrings(rms_suf);
    compare_rms_substrings<sa_index> cmp(rms_suf.text, substrings_container);

    // Sort rms-substrings (suffix-indices) inside of buckets
    size_t bucket_index;
    sa_index interval_begin, interval_end = rms_suf.relative_indices.size();
    util::span<sa_index> current_interval;
    // Sort every rms-bucket, starting at last bucket
    for (size_t first_letter = max_character_value - 1; 0 < first_letter;
         --first_letter) {
        for (size_t second_letter = max_character_value;
             first_letter < second_letter; --second_letter) {
            /*std::cout << "Currently sorting (" << first_letter << ","
                      << second_letter << ")-bucket." << std::endl;*/
            bucket_index =
                sa_buckets.get_rms_bucket_index(first_letter, second_letter);
            interval_begin = sa_buckets.s_buckets[bucket_index];
            if (interval_begin < interval_end) {
                // Interval of indices to sort
                current_interval = rms_suf.relative_indices.slice(
                    interval_begin, interval_end);
                // sort current interval/bucket
                util::sort::introsort<sa_index,
                                      compare_rms_substrings<sa_index>>(
                    current_interval, cmp);

                // Modify elements in interval if they are the same (MSB set)
                /*std::cout
                    << "Checking bucket for unsorted (i.e. same) substrings."
                    << std::endl;*/
                set_unsorted_rms_substring_intervals(
                    rms_suf, cmp, interval_begin, interval_end);
                // Refresh end_index
                interval_end = interval_begin;
            } else {
                /*std::cout << "Skipping bucket (" << first_letter << ","
                          << second_letter << ")" << std::endl;*/
            }
        }
    }
}

template <typename sa_index>
inline static void compute_initial_isa(util::span<sa_index> rel_ind,
                                       util::span<sa_index> isa) {
    sa_index sorted_count = 0, unsorted_count = 0;
    sa_index rank = rel_ind.size() - 1;
    DCHECK_EQ(rel_ind.size(), isa.size());
    std::cout << "interval of size " << rel_ind.size() << std::endl;
    for (sa_index pos = rel_ind.size(); 0 < pos; --pos) {
        std::cout << "Index: " << rel_ind[pos - 1] << ", negated: "
                  << (rel_ind[pos - 1] ^ utils<sa_index>::NEGATIVE_MASK)
                  << std::endl;
        // Current index has been negated
        if ((rel_ind[pos - 1] & utils<sa_index>::NEGATIVE_MASK) > 0) {
            rel_ind[pos - 1] ^= utils<sa_index>::NEGATIVE_MASK;
            std::cout << "Negated index found." << std::endl;
            std::cout << "Set rank for index " << rel_ind[pos - 1] << " to "
                      << rank << std::endl;
            isa[rel_ind[pos - 1]] = rank;
            ++unsorted_count;
            if (sorted_count > 0) {
                std::cout << "Sorted interval of length " << sorted_count
                          << " ending at " << pos << std::endl;
                // Could cause issues with inducing (normally inverted)
                rel_ind[pos] = sorted_count | utils<sa_index>::NEGATIVE_MASK;
                // Reset sorted count
                sorted_count = 0;
            }
        } else {
            // Current index not negated - either in interval of sorted
            // indices or interval of unsorted indices ended.
            if (unsorted_count > 0) {
                std::cout << "Beginning of unsorted interval found."
                          << std::endl;
                std::cout << "Set rank for index " << rel_ind[pos - 1] << " to "
                          << rank << std::endl;
                // Set rank for "final" index in unsorted interval (i.e.
                // interval with same substrings).
                isa[rel_ind[pos - 1]] = rank;
                // Reduce rank by unsorted_count (number of indices with
                // same rank; need to increase count one more time
                // (for current index))
                // Overflow can only happen, if first index in rel_ind is
                // part of unsorted interval.
                rank -= ++unsorted_count;
                std::cout << "Reduced rank to " << rank << std::endl;
                // Reset unsorted count;
                unsorted_count = 0;
            } else {
                std::cout << "Set rank for index " << rel_ind[pos - 1] << " to "
                          << rank << std::endl;
                isa[rel_ind[pos - 1]] = rank--;
                ++sorted_count;
            }
        }
    }
    if (sorted_count > 0) {
        std::cout << "Sorted interval of length " << sorted_count
                  << " ending at " << 0 << std::endl;
        rel_ind[0] = sorted_count | utils<sa_index>::NEGATIVE_MASK;
    }
}

/** \brief recomputes ranks (isa) for a given interval after it has been
 sorted by sort-rms-suffixes


  @param rel_ind: The interval of relative indices to recompute the isa
 for
  @param isa: The complete partial isa (for all rms-suffixes)
  @param cmp: The compare function containing the depth currently to be
 considered

  @returns Wether the interval to recompute the isa for is completely and
 uniquely sorted.
*/
template <typename sa_index>
inline static bool recompute_isa(util::span<sa_index> rel_ind_ctr,
                                 util::span<sa_index> rel_ind,
                                 util::span<sa_index> isa,
                                 compare_suffix_ranks<sa_index> cmp) {

    DCHECK_EQ(rel_ind.size(), isa.size());
    std::cout << "rel_ind (size " << rel_ind.size() << "): " << rel_ind << std::endl;

    sa_index sorted_count=0, unsorted_count=0, rank = rel_ind.size()-1;
    sa_index elem, predec;
    bool sorted = true;
    // Iterate over complete rms-suffix indices to recompute ranks.


    for(sa_index pos = rel_ind.size(); 1 < pos; --pos) {
        elem = pos-1;
        predec = pos-2;
    // Either this index or predecessor has negated index (i.e. sorted)
    // -> skip
        if ((rel_ind[elem] & utils<sa_index>::NEGATIVE_MASK) > 0) {
            DCHECK_EQ(unsorted_count, 0);
            std::cout << "Index is negated." << std::endl;
            ++sorted_count;
            // New sorted interval directly behind this one -> increase sorted length at this pos
            rel_ind[elem] = sorted_count | utils<sa_index>::NEGATIVE_MASK;
            DCHECK_EQ(isa[rel_ind_ctr[elem]], rank);
            --rank;
        }
        else if((rel_ind[predec] & utils<sa_index>::NEGATIVE_MASK) > 0) {
            std::cout << "Predecessor is negated." << std::endl;
                std::cout << "Gave index " << rel_ind[elem] << " rank " << rank
                      << std::endl;
            // Predecessor is in sorted interval -> set rank for index at elem
            if(unsorted_count == 0) {
                isa[rel_ind[elem]] = rank--;
                //Already compared it to its predecessor
                ++sorted_count;
            } else if(unsorted_count > 0) {
                /*  This case only occurs, when an unsorted interval is directly
                    behind the beginning of a sorted interval, i.e. elem is unsorted
                */
                std::cout << "Unsorted interval ended at position " << elem
                          << std::endl;
                // Still part of unsorted interval
                isa[rel_ind[elem]] = rank;
                rank -= (++unsorted_count);
                std::cout << "new rank: " << rank << std::endl;
                unsorted_count = 0;
            }
        }
        // Check if current position is sorted uniquely to its predecessor
        // if(cmp(predec, elem) > 0) {
        // If sorted: predecessor is smaller than current index
        else if (cmp(rel_ind[predec], rel_ind[elem]) > 0) {
            if (unsorted_count > 0) {
                std::cout << "Unsorted interval ended at position " << elem
                          << std::endl;
                // Still part of unsorted interval
                isa[rel_ind[elem]] = rank;
                std::cout << "Gave index " << rel_ind[elem] << " rank " << rank
                      << std::endl;
                rank -= (++unsorted_count);
                std::cout << "new rank: " << rank << std::endl;
                // Reset unsorted count (sorted interval started)
                unsorted_count = 0;
            } else {
                std::cout << "Gave index " << rel_ind[elem] << " rank " << rank
                      << std::endl;
                isa[rel_ind[elem]] = rank--;
            }
            ++sorted_count;
        } else {
            // Still unsorted:
            if (sorted_count > 0) {
                // There was an interval of sorted elements
                rel_ind[pos] = sorted_count | utils<sa_index>::NEGATIVE_MASK;
                DCHECK_EQ((rel_ind[pos] ^ utils<sa_index>::NEGATIVE_MASK), sorted_count);
                std::cout << "Sorted interval of length " << sorted_count
                          << " ended at position " << pos << " with rank " << rank << std::endl;
                // Reduce rank because elem was still sorted
                DCHECK_EQ(isa[rel_ind_ctr[pos]], rank+1);
                //rank--;
                // Reset sorted count (unsorted interval started)
                sorted_count = 0;
            }
            isa[rel_ind[elem]] = rank;
            std::cout << "Gave index " << rel_ind[elem] << " rank " << rank
                      << std::endl;
            ++unsorted_count;
            sorted = false;
        }
    }

    sa_index ref = rel_ind[0];
    if((ref & utils<sa_index>::NEGATIVE_MASK) > 0) {
        ref ^= utils<sa_index>::NEGATIVE_MASK;
        // Has already been sorted earlier, i.e. has rank 0
        if(sorted_count > 0) {
            std::cout << "last element to be refreshed (interval size)." << std::endl;
            rel_ind[ref] = (++sorted_count) | utils<sa_index>::NEGATIVE_MASK;
        }
        return sorted;
    }
    // first element in unsorted interval
    else if(unsorted_count > 0) {
        std::cout << "unsorted." << std::endl;
        // Set rank of current (unsorted) interval; rel_ind doesn't change
        isa[ref] = rank;
        std::cout << "First index of interval is part of unsorted interval "
                     "with rank "
                  << rank << std::endl;
        return false;
    }
    // first element in sorted interval
    else {
        std::cout << "sorted." << std::endl;
        // lowest possible rank for first index if part of sorted interval
        isa[ref] = 0;
        // Set rel. index to negated length of sorted interval
        rel_ind[0] = (++sorted_count) | utils<sa_index>::NEGATIVE_MASK;
        std::cout << "First index of interval is part of sorted interval "
                     "of length "
                  << sorted_count << std::endl;
        return sorted;
    }
}


// One iteration for sorting rms-suffixes
template <typename sa_index>
inline static void
sort_rms_suffixes_internal(util::span<sa_index> rel_ind_ctr, rms_suffixes<sa_index>& rms_suf,
                           compare_suffix_ranks<sa_index> cmp) {
    sa_index interval_begin = 0, interval_end = 0, current_index;
    // indicator wether unsorted interval was found (to sort)
    bool unsorted = false;
    util::span<sa_index> rel_ind = rms_suf.relative_indices;
    util::span<sa_index> isa = rms_suf.partial_isa;

    for (sa_index pos = 0; pos < rms_suf.partial_isa.size(); ++pos) {
        if(pos == rms_suf.partial_isa.size()-1) {
            interval_end = rms_suf.partial_isa.size();
        }
        // Search for unsorted interval
        current_index = rms_suf.relative_indices[pos];
        // Negated value
        if ((current_index & utils<sa_index>::NEGATIVE_MASK) > 0) {
            if(unsorted > 0) {std::cout << "End of unsorted interval found at: " << pos
            << std::endl;}
            // End of unsorted interval found
            interval_end = pos;
            // Skip interval of sorted elements (negated length contained in
            // current_index)
            pos += (current_index ^ utils<sa_index>::NEGATIVE_MASK);
        } else {
            if(unsorted == 0 && pos > 0) {
            std::cout << "Start of unsorted interval set to " << pos-1
                      << std::endl;
            interval_begin = pos-1;
            }
            unsorted = true;
        }
        // if unsorted interval contains more than one element (after
        // interval_end has been set)
        if (unsorted > 0 && interval_end > interval_begin) {
            std::cout << "Sorting unsorted interval from " << interval_begin
                      << " to " << interval_end << std::endl;
            auto slice_to_sort = rel_ind.slice(interval_begin, interval_end);
            std::cout << "Slice to sort of length " << slice_to_sort.size()
                      << std::endl;
            util::sort::introsort<sa_index, compare_suffix_ranks<sa_index>>(
                rel_ind.slice(interval_begin, interval_end), cmp);
            // Reset indicator
            std::cout << "Unsorted indicator reset." << std::endl;
            unsorted = false;
            // Refresh ranks for complete isa
            std::cout << "Recomputing ranks." << std::endl;
            recompute_isa(rel_ind_ctr, rel_ind, isa, cmp);
        }
    }
}

template <typename sa_index>
inline static void sort_rms_suffixes(util::span<sa_index> rel_ind_ctr, rms_suffixes<sa_index>& rms_suf) {
    sa_index pos;
    compare_suffix_ranks<sa_index> cmp(rms_suf.partial_isa, 0);
    bool unsorted = true;
    util::span<sa_index> rel_ind = rms_suf.relative_indices;
    // At most that many iterations (if we have to consider last suffix (or
    // later))
    sa_index max_iterations =
        rms_suf.relative_indices.size();
        //util::floor_log2(rms_suf.relative_indices.size()) + 1;
    // while(unsorted) {
    for (sa_index iter = 0; iter < max_iterations + 1; ++iter) {
        std::cout << "__________________________" << std::endl;
        std::cout << "Iteration " << iter + 1 << " | Calling internal sorting."
                  << std::endl;
        // Sort rms-suffixes
        sort_rms_suffixes_internal(rel_ind_ctr, rms_suf, cmp);
        std::cout << "Internal sorting finished" << std::endl;
        std::cout << "__________________________" << std::endl;

        unsorted = false;
        // Check if still unsorted:
        pos = 0;
        sa_index ref;
        while (pos < rel_ind.size()) {
            // Negated length of sorted interval
            if ((rel_ind[pos] & utils<sa_index>::NEGATIVE_MASK) > 0) {
                // Negated index at pos
                ref = (rel_ind[pos] ^ utils<sa_index>::NEGATIVE_MASK);
                std::cout << ref << std::endl;
                DCHECK_NE(ref, 0);
                pos += (rel_ind[pos] ^ utils<sa_index>::NEGATIVE_MASK);
                std::cout << "Sorted interval found. Skipped to " << pos
                          << std::endl;
            } else {
                // interval beginning with non-negated index found ->
                // unsorted interval
                std::cout << "Interval not sorted (starting at pos " << pos
                          << ")- still non-unique substrings checked."
                          << std::endl;
                unsorted = true;
                // End inner loop - we know that there are still unsorted
                // intervals
                break;
            }
        }
        // Everything has been sorted - break outer loop
        if (unsorted == 0) {
            std::cout << "No unsorted interval left - ending "
                         "rms-suffix-sorting routine."
                      << std::endl;
            break;
        }
        // Increase depth for next iteration
        ++cmp.depth;
        std::cout << "Depth currently at " << cmp.depth << std::endl;
    }
}

/** \brief Sorts all rms-suffixes at the beginning of out_sa in their
 correct order (via precomputed ranks)

*/
template <typename sa_index>
inline static void sort_rms_indices_to_order(rms_suffixes<sa_index>& rms_suf,
                                             sa_index rms_count,
                                             util::container<bool>& types,
                                             util::span<sa_index> out_sa) {
    sa_index correct_pos;
    std::cout << "initial rms-count " << rms_count << std::endl;
    auto isa = rms_suf.partial_isa;

    std::cout << "isa: " << isa << std::endl;

    // Skip last index because of sentinel
    for (sa_index pos = rms_suf.text.size() - 1; 1 < pos; --pos) {
        correct_pos = pos - 1;
        // RMS-Suffix in text found
        if (sa_types<sa_index>::is_rms_type(correct_pos, types)) {

            // If predecessor of correct_pos is l-type: negate, because not
            // considered in first induce step
            if (sa_types<sa_index>::is_l_type(correct_pos - 1, types)) {
                std::cout << "Predecessor of " << correct_pos
                          << " is l-type suffix -> negate index." << std::endl;
                out_sa[isa[--rms_count]] =
                    correct_pos ^ utils<sa_index>::NEGATIVE_MASK;
                std::cout << "Index " << out_sa[isa[rms_count]]
                          << " at position " << isa[rms_count] << std::endl;
            } else {
                // Index correct_pos considered in first induce step
                std::cout << "Predecessor of " << correct_pos
                          << " is no l-type suffix -> non-negated index."
                          << std::endl;
                out_sa[isa[--rms_count]] = correct_pos;
                std::cout << "Index " << out_sa[isa[rms_count]]
                          << " at position " << isa[rms_count] << std::endl;
            }
            // out_sa[isa[--rms_count]] =
            // (sa_types<sa_index>::is_l_type(correct_pos-1, types)) ?
            // current_pos ^ utils<sa_index>::NEGATIVE_MASK : current_pos;
        }
    }
    if (sa_types<sa_index>::is_rms_type(0, types)) {
        DCHECK_GT(rms_count, 0);

        out_sa[isa[--rms_count]] = 0;
        std::cout << "Index " << out_sa[isa[rms_count]] << " at position "
                  << isa[rms_count] << std::endl;
    }
}

/** \brief Sorts all rms-suffixes, according to their order from out_sa[0,
 rms_count) into their correct positions in out_sa

*/
template <typename sa_index>
inline static void insert_rms_into_correct_pos(sa_index rms_count,
                                               buckets<sa_index>& bkts,
                                               const size_t max_character_code,
                                               util::span<sa_index> out_sa) {
    sa_index s_bkt_index =
        bkts.get_s_bucket_index(max_character_code, max_character_code);
    sa_index right_border_s, right_border_rms;
    bkts.s_buckets[s_bkt_index] =
        out_sa.size() - 1; // Last pos for last bkt (never used)
    for (size_t c0 = max_character_code - 1; 0 < c0; --c0) {
        // new border one pos left of next l-bucket (i.e. for c0+1)
        right_border_s = bkts.l_buckets[c0 + 1] - 1;
        for (size_t c1 = max_character_code; c0 < c1; --c1) {
            s_bkt_index = bkts.get_s_bucket_index(c0, c1);
            right_border_rms = right_border_s - bkts.s_buckets[s_bkt_index];
            // Set bucket value to right border of (c0,c1)-s-bucket
            std::cout << "Right border for s-bucket (" << c0 << "," << c1
                      << "): " << right_border_s << std::endl;
            bkts.s_buckets[s_bkt_index] = right_border_s;
            // Compute bucket index for current rms-bucket
            s_bkt_index = bkts.get_rms_bucket_index(c0, c1);
            std::cout << "Inserting rms-suffixes into correct position. "
                         "Starting at "
                      << right_border_rms << std::endl;
            // Only insert rms-suffixes if current rms-count is smaller than
            // computed (right) border for rms-suffixes in (c0,c1)-bucket
            for (right_border_s = right_border_rms;
                 bkts.s_buckets[s_bkt_index] < rms_count; --right_border_s) {
                // Insert all rms-suffixes corresponding to this bucket
                // All indices of rms-suffixes contained in corresponding
                // order in out_sa[0,rms_count)
                out_sa[right_border_s] = out_sa[--rms_count];
                std::cout << "Inserted index " << out_sa[right_border_s]
                          << " at pos " << right_border_s << std::endl;
            }
        }
        // Index for s-bucket (c0,c0)
        s_bkt_index = bkts.get_s_bucket_index(c0, c0);
        // Contains first position for interval to check in first induce
        // step; bkts.s_buckets[s_bkt_index] has yet to be altered (i.e.
        // contains number of s-suffixes for (c0,c0)-bucket)
        bkts.s_buckets[bkts.get_rms_bucket_index(c0, c0 + 1)] =
            right_border_s - bkts.s_buckets[s_bkt_index] + 1;
        std::cout << "End border for inducing for (" << c0 << ",*)-bucket: "
                  << bkts.s_buckets[bkts.get_rms_bucket_index(c0, c0 + 1)]
                  << std::endl;
        // Refresh border for (c0,c0)-bucket to point to most right position
        // of bucket (not part of inner loop)
        std::cout << "Right border for s-bucket " << c0 << "," << c0 << ": "
                  << right_border_s << std::endl;
        bkts.s_buckets[s_bkt_index] = right_border_s;
    }
}

} // namespace sacabench::div_suf_sort
