#pragma once

#include "comparison_fct.hpp"
#include "utils.hpp"
#include <queue>
#include <util/assertions.hpp>
#include <util/bits.hpp>
#include <util/container.hpp>
#include <util/sort/introsort.hpp>
#include <util/sort/std_sort.hpp>
#include <util/span.hpp>

namespace sacabench::div_suf_sort {

template <typename sa_index>
inline static void set_unsorted_rms_substring_intervals(
    rms_suffixes<sa_index>& rms_suf, compare_rms_substrings<sa_index> cmp,
    size_t interval_start, size_t interval_end) {
    DCHECK_LT(interval_start, interval_end - 1);
    size_t elem, compare_to;
    // Last element to be compared to its predecessor: interval_start + 1
    // interval_end must not contain last element of interval
    // bool less, greater;
    for (size_t pos = interval_end - 1; interval_start < pos; --pos) {
        elem = rms_suf.relative_indices[pos - 1];
        compare_to = rms_suf.relative_indices[pos];
        DCHECK_EQ(cmp(compare_to, elem), false);
        // std::cout << "comparing " << elem << " to " << compare_to <<
        // std::endl;
        // None of the substrings is smaller than the other (i.e. the same)
        if (!cmp(elem, compare_to)) {
            rms_suf.relative_indices[pos] =
                rms_suf.relative_indices[pos] | utils<sa_index>::NEGATIVE_MASK;
        }
    }
}

template <typename sa_index>
inline static void sort_rms_substrings(rms_suffixes<sa_index>& rms_suf,
                                       const size_t max_character_value,
                                       buckets<sa_index>& sa_buckets) {

    // Compute RMS-Substrings (tupel with start-/end-position)
    auto substrings_container = extract_rms_substrings(rms_suf);
    compare_rms_substrings<sa_index> cmp(rms_suf.text, substrings_container);

    // Sort rms-substrings (suffix-indices) inside of buckets
    size_t bucket_index;
    size_t interval_begin, interval_end = rms_suf.relative_indices.size();
    util::span<sa_index> current_interval;
    // Sort every rms-bucket, starting at last bucket
    for (size_t first_letter = max_character_value - 1; 0 < first_letter;
         --first_letter) {
        for (size_t second_letter = max_character_value;
             first_letter < second_letter; --second_letter) {
            bucket_index =
                sa_buckets.get_rms_bucket_index(first_letter, second_letter);
            interval_begin = sa_buckets.s_buckets[bucket_index];
            if (interval_begin < interval_end) {
                // Interval of indices to sort
                current_interval = rms_suf.relative_indices.slice(
                    interval_begin, interval_end);
                if (current_interval.size() > 1) {
                    // sort current interval/bucket
                    // std::cout << "sorting interval." << std::endl;
                    util::sort::introsort<sa_index,
                                          compare_rms_substrings<sa_index>>(
                        current_interval, cmp);
                    // std::cout << "sorting interval complete." << std::endl;

                    // Modify elements in interval if they are the same (MSB
                    // set)
                    set_unsorted_rms_substring_intervals(
                        rms_suf, cmp, interval_begin, interval_end);
                }
                // Refresh end_index
                interval_end = interval_begin;
            }
        }
    } /*
     interval_end = rms_suf.relative_indices.size();
     current_interval = rms_suf.relative_indices.slice(
                     interval_begin, interval_end);
     if(current_interval.size() > 1) {
         util::sort::introsort<sa_index,
                             compare_rms_substrings<sa_index>>(
             current_interval, cmp);

         // Modify elements in interval if they are the same (MSB
         // set)
         set_unsorted_rms_substring_intervals(
             rms_suf, cmp, interval_begin, interval_end);
     }*/
}

template <typename sa_index>
inline static void compute_initial_isa(util::span<sa_index> rel_ind,
                                       util::span<sa_index> isa) {
    size_t sorted_count = 0, unsorted_count = 0;
    size_t rank = rel_ind.size() - 1;
    DCHECK_EQ(rel_ind.size(), isa.size());
    for (size_t pos = rel_ind.size(); 0 < pos; --pos) {
        // Current index has been negated
        if ((rel_ind[pos - 1] & utils<sa_index>::NEGATIVE_MASK) > 0) {
            if (sorted_count > 0) {
                DCHECK_EQ(unsorted_count, 0);
                // Could cause issues with inducing (normally inverted)
                rel_ind[pos] = sorted_count | utils<sa_index>::NEGATIVE_MASK;
                // Reset sorted count
                sorted_count = 0;
            }
            rel_ind[pos - 1] =
                rel_ind[pos - 1] ^ utils<sa_index>::NEGATIVE_MASK;
            isa[rel_ind[pos - 1]] = rank;
            ++unsorted_count;

        } else {
            // Current index not negated - either in interval of sorted
            // indices or interval of unsorted indices ended.
            if (unsorted_count > 0) {
                DCHECK_EQ(sorted_count, 0);
                // Set rank for "final" index in unsorted interval (i.e.
                // interval with same substrings).
                isa[rel_ind[pos - 1]] = rank;
                /*  Reduce rank by unsorted_count (number of indices with
                    same rank; need to increase count one more time
                    (for current index))
                    Overflow can only happen, if first index in rel_ind is
                    part of unsorted interval.*/
                rank -= (++unsorted_count);
                // Reset unsorted count;
                unsorted_count = 0;
            } else {
                isa[rel_ind[pos - 1]] = rank--;
                ++sorted_count;
            }
        }
    }
    if (sorted_count > 0) {
        rel_ind[0] = sorted_count | utils<sa_index>::NEGATIVE_MASK;
    }
}

template <typename sa_index>
inline static bool
recompute_interval_isa(util::span<sa_index> rel_ind, size_t interval_begin,
                       size_t interval_end, util::span<sa_index> isa,
                       compare_suffix_ranks<sa_index> cmp) {
    DCHECK_LT(interval_begin, interval_end);
    DCHECK_LE(interval_end, rel_ind.size());
    // Variables to check, wether interval is sorted at either the beginning or
    // the end of the interval (to readjust sorted-interval sizes of
    // predecessors/successors)
    // size_t sorted_size_begin=0;
    size_t sorted_size_end = 0;

    bool current_sorted = true, is_sorted = true;
    size_t sorted_begin = interval_begin, unsorted_begin = interval_begin,
           rank = interval_begin;
    sa_index current, next;

    std::queue<sa_index> elements;

    // Recompute ranks, skip last element of interval
    for (size_t pos = interval_begin; pos < interval_end - 1; ++pos) {
        current = rel_ind[pos];
        elements.push(current);
        next = rel_ind[pos + 1];
        DCHECK_LE(isa[current], isa[next]);
        // All indices non-negated
        if (cmp(current, next)) {
            if (!current_sorted) {
                sorted_begin = pos + 1;
                current_sorted = true;
                // Set ranks for unsorted interval (highest rank for each of
                // them)
                for (size_t i = unsorted_begin; i < pos + 1; ++i) {
                    isa[rel_ind[i]] = rank;
                }
            } else {
                isa[current] = rank;
            }
            // Increase rank after this step
            ++rank;
        } else {
            if (is_sorted) {
                is_sorted = false;
            }
            if (current_sorted) {
                unsorted_begin = pos;
                current_sorted = false;
                // Correct size, because pos is already part of unsorted
                // interval
                // May occur if first element of interval is unsorted
                if (pos - sorted_begin > 0) {
                    rel_ind[sorted_begin] =
                        (pos - sorted_begin) ^ utils<sa_index>::NEGATIVE_MASK;
                }

                if (sorted_begin == interval_begin) {
                    // sorted_size_begin = pos - sorted_begin;
                }
            }
            ++rank;
        }
    }
    elements.push(rel_ind[interval_end - 1]);
    DCHECK_EQ(rank, interval_end - 1);
    // Handle last element of interval
    if (current_sorted) {
        if (sorted_begin == interval_begin) {
            // sorted_size_begin = sorted_begin;
        }
        // std::cout << "Last sorted interval." << std::endl;
        sorted_size_end = interval_end - sorted_begin;
        DCHECK_GT(sorted_size_end, 0);
        isa[rel_ind[interval_end - 1]] = rank;
        rel_ind[sorted_begin] =
            sorted_size_end | utils<sa_index>::NEGATIVE_MASK;
    } else {
        // std::cout << "last unsorted interval" << std::endl;
        // Part of unsorted interval -> set ranks
        for (size_t i = unsorted_begin; i < interval_end; ++i) {
            isa[rel_ind[i]] = rank;
        }
    }
    // Refresh ranks for cmp fct (after they have been completely refreshed
    // in isa for current interval)
    DCHECK_EQ(elements.size(), (interval_end - interval_begin));
    for (size_t i = 0; i < (interval_end - interval_begin); ++i) {
        current = elements.front();
        elements.pop();
        cmp.partial_isa[current] = isa[current];
    }

    /*
    // Alternatively: Search from beginning, until this interval is found
    size_t length;
    // Check, wether previous interval is sorted
    if (sorted_size_begin > 0) {
        std::cout << "Sorted interval at beginning." << std::endl;
        sorted_begin = interval_begin;
        for (size_t pos = interval_begin; 0 < pos; --pos) {
            std::cout << "loop pos " << pos << std::endl;
            current = rel_ind[pos - 1];
            next = rel_ind[pos];
            if ((current & utils<sa_index>::NEGATIVE_MASK) > 0) {
                length = (current ^ utils<sa_index>::NEGATIVE_MASK);
                // Either sorted length of preceding sorted interval is
                //increased or we can stop here
                std::cout << pos - 1 + length << std::endl;
                if (pos - 1 + length == sorted_begin) {
                    // Refresh sorted begin to search for further sorted
                    // intervals
                        // in front (directly connected to this one)
                        sorted_begin = pos - 1;
                    // set sorted size begin to reflect current completely
                    // sorted interval
                    sorted_size_begin += length;
                    rel_ind[pos - 1] =
                        sorted_size_begin | utils<sa_index>::NEGATIVE_MASK;
                } else {
                    std::cout << "This maybe shouldn't have occured."
                                << std::endl;
                    break;
                }

            } else if ((next & utils<sa_index>::NEGATIVE_MASK) > 0) {
                // Prev index was beginning of sorted interval.
                std::cout <<
                          "Prev index was beginning of sorted interval -> Skip"
                          << std::endl;
            }
            // There is an unsorted interval before a sorted interval
            //"indicator"
            else if (isa[current] == isa[next]) {
                std::cout << "break me up" << std::endl;
                break;
            }
        }
    }
    // There is a sorted interval at the end (not at end of rel_ind) and there
    // is a sorted interval directly after this one
        if (sorted_size_end > 0 && interval_end < rel_ind.size() &&
             (rel_ind[interval_end] & utils<sa_index>::NEGATIVE_MASK) > 0) {
        // Set correct length of last sorted interval in this interval
        rel_ind[sorted_begin] =
            sorted_size_end +
            (rel_ind[interval_end] ^ utils<sa_index>::NEGATIVE_MASK);
    }*/

    return is_sorted;
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
inline static bool recompute_isa_ltr(util::span<sa_index> rel_ind,
                                     util::span<sa_index> isa,
                                     compare_suffix_ranks<sa_index> cmp) {

    //
    DCHECK_EQ(rel_ind.size(), isa.size());
    size_t sorted_begin = 0, unsorted_begin = 0, rank = 0;
    sa_index current, next;
    bool is_sorted = true, current_sorted = false;

    for (size_t pos = 0; pos < rel_ind.size() - 1; ++pos) {
        DCHECK_LT(rank, rel_ind.size());
        current = rel_ind[pos];
        next = rel_ind[pos + 1];
        if ((current & utils<sa_index>::NEGATIVE_MASK) > 0) {
            // Not set in this iteration; still needed to be increased
            // Set sorted_begin if this is a new sorted interval beginning
            if (!current_sorted) {
                sorted_begin = pos;
                current_sorted = true;
            }
            // current contains negated length of sorted interval -> skip
            current = current ^ utils<sa_index>::NEGATIVE_MASK;

            // Counter ++pos in loop header
            pos += current - 1;
            rank += current;
        } else {
            if ((next & utils<sa_index>::NEGATIVE_MASK) > 0) {
                // Next index is part of sorted interval;
                if (!current_sorted) {
                    // Set rank for all elements in unsorted interval
                    // (from unsorted_begin to current)
                    for (size_t i = unsorted_begin; i < pos + 1; ++i) {
                        isa[rel_ind[i]] = rank;
                    }
                } else {
                    isa[current] = rank;
                }
                ++rank;
            } else if (!cmp(current, next)) {

                // (current, next) not sorted correctly -> unsorted interval
                is_sorted = false;
                if (current_sorted) {
                    // Unsorted interval starting at pos
                    current_sorted = false;
                    unsorted_begin = pos;
                    // Set length for (previous) sorted interval
                    // If two unsorted intervals follow each other - skip this
                    // operation
                    if (pos - sorted_begin > 0) {
                        rel_ind[sorted_begin] = (pos - sorted_begin) ^
                                                utils<sa_index>::NEGATIVE_MASK;
                    }
                }
                ++rank;
            } else {
                // Set correct rank for sorted element

                // (current, next) sorted correctly
                if (!current_sorted) {
                    // isa[current] = rank;
                    current_sorted = true;
                    // Unsorted intervals could have only occured until now, if
                    // pos > 0
                    // Condition needed, because current_sorted ininitialized
                    // with false (0)
                    if (pos > 0) {
                        // sorted interval always starts after(!) current pos
                        // Only exception: pos 0 (doesn't matter, because
                        // sorted_begin was initialized correctly)
                        sorted_begin = pos + 1;
                        // Unsorted interval ended.
                        // Set rank for all elements in unsorted interval
                        // (from unsorted_begin to current)
                        for (size_t i = unsorted_begin; i < pos + 1; ++i) {
                            isa[rel_ind[i]] = rank;
                        }
                    } else {
                        isa[current] = rank;
                    }
                    ++rank;
                } else {
                    isa[current] = rank++;
                }
            }
        }
    }

    current = rel_ind[rel_ind.size() - 1];
    // First case necessary?
    if ((current & utils<sa_index>::NEGATIVE_MASK) > 0) {
        if (current_sorted) {
            rel_ind[sorted_begin] = (rel_ind.size() - sorted_begin) |
                                    utils<sa_index>::NEGATIVE_MASK;
        }
    } else if (current_sorted) {
        rel_ind[sorted_begin] =
            (rel_ind.size() - sorted_begin) | utils<sa_index>::NEGATIVE_MASK;
    } else {
        for (size_t i = unsorted_begin; i < rel_ind.size(); ++i) {
            isa[rel_ind[i]] = rank;
        }
    }
    return is_sorted;
}

// One iteration for sorting rms-suffixes
template <typename sa_index>
inline static bool
sort_rms_suffixes_internal(rms_suffixes<sa_index>& rms_suf,
                           compare_suffix_ranks<sa_index> cmp) {
    size_t interval_begin = 0, interval_end = 0;

    sa_index current_index = 0, next_index = 0;
    // indicator wether unsorted interval was found (to sort)
    bool unsorted = false, current_unsorted = false;
    util::span<sa_index> rel_ind = rms_suf.relative_indices;
    util::span<sa_index> isa = rms_suf.partial_isa;

    for (size_t pos = 0; pos < isa.size(); ++pos) {

        // Last index: set interval_end for possible sort up to last element.
        // May be overwritten, if last element is its own (sorted) bucket
        if (pos == isa.size() - 1) {
            interval_end = isa.size();
        }
        // Search for unsorted interval
        current_index = rel_ind[pos];
        if (pos < isa.size() - 1) {
            next_index = rel_ind[pos + 1];
        }
        // Negated value
        if ((current_index & utils<sa_index>::NEGATIVE_MASK) > 0) {
            // End of unsorted interval found
            interval_end = pos;
            // Skip interval of sorted elements (negated length contained in
            // current_index)
            pos += ((current_index ^ utils<sa_index>::NEGATIVE_MASK) - 1);

        } else if (!current_unsorted) {
            interval_begin = pos;
            current_unsorted = true;
        } else if (pos < isa.size() - 1 &&
                   (next_index & utils<sa_index>::NEGATIVE_MASK) == 0 &&
                   isa[current_index] < isa[next_index]) {
            // Rare case where there are two unsorted intervals directly next
            // to each other.
            // Limit to first interval.
            interval_end = pos + 1;
        }
        // if unsorted interval contains more than one element (after
        // interval_end has been set)
        // In last iteration interval_end will always be set before this cond.!
        if (current_unsorted > 0 && interval_end > interval_begin) {
            if (interval_end - interval_begin > 1) {
                if (unsorted == 0) {
                    unsorted = true;
                }
                util::sort::introsort<sa_index, compare_suffix_ranks<sa_index>>(
                    rel_ind.slice(interval_begin, interval_end), cmp);
            }
            recompute_interval_isa(rel_ind, interval_begin, interval_end, isa,
                                   cmp);

            // Refresh ranks for complete isa
            // recompute_isa_ltr(rel_ind, isa, cmp);

            // Reset indicator
            current_unsorted = false;
        }
    }

    return unsorted;
}

template <typename sa_index>
inline static void sort_rms_suffixes(rms_suffixes<sa_index>& rms_suf) {
    // Copy content of partial isa into isa_cmp (isa particular for cmp fct)
    // Copy needed for specific cases while recomputing ranks (referencing
    // through depth points to indices in same unsorted interval)
    util::span<sa_index> rel_ind = rms_suf.relative_indices;
    util::span<sa_index> isa = rms_suf.partial_isa;
    auto isa_cmp = util::make_container<sa_index>(isa.size());
    DCHECK_EQ(isa_cmp.size(), isa.size());
    for (size_t i = 0; i < isa_cmp.size(); ++i) {
        isa_cmp[i] = isa[i];
        DCHECK_EQ(isa_cmp[i], isa[i]);
    }
    compare_suffix_ranks<sa_index> cmp(isa_cmp, 0);
    bool unsorted;
    // At most that many iterations (if we have to consider last suffix (or
    // later))
    size_t max_iterations = rms_suf.relative_indices.size();

    size_t depth;
    size_t max_index;
    sa_index current, next;

    // TODO (optimization): Check if max_iterations can be upper bounded
    // util::floor_log2(rms_suf.relative_indices.size()) + 1;
    for (size_t iter = 0; iter < max_iterations + 1; ++iter) {
        // Sort rms-suffixes
        for (size_t pos = 0; pos < rel_ind.size() - 1; ++pos) {
            current = rel_ind[0];
            next = rel_ind[1];

            // Neither is negated, i.e. beginning of sorted interval
            if (!((current & utils<sa_index>::NEGATIVE_MASK) > 0 ||
                  (next & utils<sa_index>::NEGATIVE_MASK) > 0)) {
                if (cmp(current, next)) {
                    // Rank of earlier element should be smaller
                    DCHECK_LT(isa[current], isa[next]);

                } else {
                    depth = pow(2, cmp.depth);
                    max_index = std::min(rel_ind.size() - current,
                                         rel_ind.size() - next);
                    max_index = std::min(max_index, depth);
                    for (sa_index offset = 0; offset < max_index; ++offset) {
                        DCHECK_EQ(isa[(current + offset)],
                                  isa[(next + offset)]);
                    }
                }
            }
        }
        unsorted = sort_rms_suffixes_internal(rms_suf, cmp);
        for (size_t i = 0; i < isa.size(); ++i) {
            DCHECK_EQ(isa[i], cmp.partial_isa[i]);
        }
        // Everything has been sorted - break outer loop
        if (unsorted == 0) {
            // Check wether each rank is unique
            util::container<sa_index> ranks =
                util::make_container<sa_index>(isa.size());
            for (size_t i = 0; i < ranks.size(); ++i) {
                DCHECK_LT(ranks[isa[i]], sa_index(1));
                ++ranks[isa[i]];
            }
            break;
        }
        // Increase depth for next iteration
        ++cmp.depth;
    }
}

/** \brief Sorts all rms-suffixes at the beginning of out_sa in their
 correct order (via precomputed ranks)

*/
template <typename sa_index>
inline static void sort_rms_indices_to_order(rms_suffixes<sa_index>& rms_suf,
                                             size_t rms_count,
                                             util::container<bool>& types,
                                             util::span<sa_index> out_sa) {
    auto isa = rms_suf.partial_isa;

    // Skip last index because of sentinel
    for (size_t pos = rms_suf.text.size() - 2; 0 < pos; --pos) {
        // RMS-Suffix in text found
        if (sa_types::is_rms_type(pos, types)) {

            // If predecessor of pos is l-type: negate, because not
            // considered in first induce step
            if (sa_types::is_l_type(pos - 1, types)) {
                out_sa[isa[--rms_count]] = pos ^ utils<sa_index>::NEGATIVE_MASK;
            } else {
                // Current index considered in first induce step
                out_sa[isa[--rms_count]] = pos;
            }
        }
    }
    if (sa_types::is_rms_type(0, types)) {
        DCHECK_EQ(rms_count - 1, 0);

        out_sa[isa[--rms_count]] = 0;
    }
}

/** \brief Sorts all rms-suffixes, according to their order from out_sa[0,
 rms_count) into their correct positions in out_sa

*/
template <typename sa_index>
inline static void insert_rms_into_correct_pos(size_t rms_count,
                                               buckets<sa_index>& bkts,
                                               const size_t max_character_code,
                                               util::span<sa_index> out_sa) {
    size_t s_bkt_index =
        bkts.get_s_bucket_index(max_character_code, max_character_code);
    size_t right_border_s, right_border_rms;
    bkts.s_buckets[s_bkt_index] =
        out_sa.size() - 1; // Last pos for last bkt (never used)
    for (size_t c0 = max_character_code - 1; 0 < c0; --c0) {
        // new border one pos left of next l-bucket (i.e. for c0+1)
        right_border_s = bkts.l_buckets[c0 + 1] - sa_index(1);
        for (size_t c1 = max_character_code; c0 < c1; --c1) {
            s_bkt_index = bkts.get_s_bucket_index(c0, c1);
            right_border_rms = right_border_s - bkts.s_buckets[s_bkt_index];
            // Set bucket value to right border of (c0,c1)-s-bucket
            bkts.s_buckets[s_bkt_index] = right_border_s;
            // Compute bucket index for current rms-bucket
            s_bkt_index = bkts.get_rms_bucket_index(c0, c1);
            // Only insert rms-suffixes if current rms-count is smaller than
            // computed (right) border for rms-suffixes in (c0,c1)-bucket
            for (right_border_s = right_border_rms;
                 bkts.s_buckets[s_bkt_index] < rms_count; --right_border_s) {
                // Insert all rms-suffixes corresponding to this bucket
                // All indices of rms-suffixes contained in corresponding
                // order in out_sa[0,rms_count)
                out_sa[right_border_s] = out_sa[--rms_count];
            }
        }
        // Index for s-bucket (c0,c0)
        s_bkt_index = bkts.get_s_bucket_index(c0, c0);
        // Contains first position for interval to check in first induce
        // step; bkts.s_buckets[s_bkt_index] has yet to be altered (i.e.
        // contains number of s-suffixes for (c0,c0)-bucket)
        bkts.s_buckets[bkts.get_rms_bucket_index(c0, c0 + 1)] =
            right_border_s - bkts.s_buckets[s_bkt_index] + 1;
        // Refresh border for (c0,c0)-bucket to point to most right position
        // of bucket (not part of inner loop)
        bkts.s_buckets[s_bkt_index] = right_border_s;
    }
}

} // namespace sacabench::div_suf_sort
