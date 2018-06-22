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
    size_t interval_start, size_t interval_end) {
    size_t elem, compare_to;
    // Last element to be compared to its predecessor: interval_start + 1
    // interval_end must not contain last element of interval
    // bool less, greater;
    for (size_t pos = interval_end - 1; interval_start < pos; --pos) {
        elem = rms_suf.relative_indices[pos - 1];
        compare_to = rms_suf.relative_indices[pos];
        // None of the substrings is smaller than the other (i.e. the same)
        if (!(cmp(elem, compare_to) || cmp(compare_to, elem))) {
            rms_suf.relative_indices[pos] = rms_suf.relative_indices[pos] | utils<sa_index>::NEGATIVE_MASK;
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
                if(current_interval.size() > 1) {
                    // sort current interval/bucket
                    util::sort::introsort<sa_index,
                                          compare_rms_substrings<sa_index>>(
                        current_interval, cmp);
                    
                    // Modify elements in interval if they are the same (MSB set)
                    set_unsorted_rms_substring_intervals(
                        rms_suf, cmp, interval_begin, interval_end);
                }
                // Refresh end_index
                interval_end = interval_begin;
            }
        }
    }
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
                // Could cause issues with inducing (normally inverted)
                rel_ind[pos] = sorted_count | utils<sa_index>::NEGATIVE_MASK;
                // Reset sorted count
                sorted_count = 0;
            }
            rel_ind[pos - 1] = rel_ind[pos - 1] ^ utils<sa_index>::NEGATIVE_MASK;
            isa[rel_ind[pos - 1]] = rank;
            ++unsorted_count;

        } else {
            // Current index not negated - either in interval of sorted
            // indices or interval of unsorted indices ended.
            if (unsorted_count > 0) {
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
    size_t current, next, sorted_begin=0, unsorted_begin=0, rank=0;
    
    bool is_sorted = true, current_sorted = false;
    
    for(size_t pos=0; pos < rel_ind.size()-1; ++pos) {
        DCHECK_LT(rank, rel_ind.size());
        current = rel_ind[pos];
        next = rel_ind[pos+1];
        if((current & utils<sa_index>::NEGATIVE_MASK) > 0) {
            // Not set in this iteration; still needed to be increased
            // Set sorted_begin if this is a new sorted interval beginning
            if(!current_sorted) {
                sorted_begin = pos;
                current_sorted = true;
            }
            // current contains negated length of sorted interval -> skip
            current ^= utils<sa_index>::NEGATIVE_MASK;
            
            // Counter ++pos in loop header
            pos += current-1;
            rank += current;
        } else {
            if((next & utils<sa_index>::NEGATIVE_MASK) > 0) {
                // Next index is part of sorted interval;
                if(!current_sorted) {
                    // Set rank for all elements in unsorted interval 
                    // (from unsorted_begin to current)
                    for(size_t i = unsorted_begin; i < pos+1; ++i) {
                        isa[rel_ind[i]] = rank;
                    }              
                } else {
                    isa[current] = rank;
                }
                ++rank;
            } else if (cmp(current, next) == 0) {
                // (current, next) not sorted correctly -> unsorted interval
                is_sorted = false;
                if(current_sorted) {
                    // Unsorted interval starting at pos
                    current_sorted = false;
                    unsorted_begin = pos;
                    // Set length for (previous) sorted interval
                    // If two unsorted intervals follow each other - skip this
                    // operation
                    if(pos - sorted_begin > 0) {
                        rel_ind[sorted_begin] = (pos - sorted_begin) ^ 
                        utils<sa_index>::NEGATIVE_MASK;
                    }
                }
                ++rank;
            } else {  
                // Set correct rank for sorted element

                // (current, next) sorted correctly
                if(!current_sorted) {
                    // isa[current] = rank;
                    current_sorted = true;
                    // Unsorted intervals could have only occured until now, if
                    // pos > 0
                    // Condition needed, because current_sorted ininitialized with
                    // false (0)
                    if(pos > 0) {
                        // sorted interval always starts after(!) current pos
                        // Only exception: pos 0 (doesn't matter, because sorted_begin
                        // was initialized correctly)
                        sorted_begin = pos+1;
                        // Unsorted interval ended.
                        // Set rank for all elements in unsorted interval 
                        // (from unsorted_begin to current)
                        for(size_t i = unsorted_begin; i < pos+1; ++i) {
                            isa[rel_ind[i]] = rank;
                        }               
                    } else {                
                        isa[current] = rank;
                    }
                    ++rank;
                }
                else {                
                    isa[current] = rank++;
                }
            }
        }
    }

    if(current_sorted) {
        rel_ind[sorted_begin] = (rel_ind.size() - sorted_begin) | 
        utils<sa_index>::NEGATIVE_MASK;
    } else {
        for(size_t i = unsorted_begin; i < rel_ind.size(); ++i) {
            isa[rel_ind[i]] = rank;
        }
    }
    
    
    return is_sorted;
}

// One iteration for sorting rms-suffixes
template <typename sa_index>
inline static void
sort_rms_suffixes_internal(rms_suffixes<sa_index>& rms_suf,
                           compare_suffix_ranks<sa_index> cmp) {
    size_t interval_begin = 0, interval_end = 0, current_index;
    // indicator wether unsorted interval was found (to sort)
    bool unsorted = false;
    util::span<sa_index> rel_ind = rms_suf.relative_indices;
    util::span<sa_index> isa = rms_suf.partial_isa;

    for (size_t pos = 0; pos < rms_suf.partial_isa.size(); ++pos) {
        if(pos == rms_suf.partial_isa.size()-1) {
            interval_end = rms_suf.partial_isa.size();
        }
        // Search for unsorted interval
        current_index = rms_suf.relative_indices[pos];
        // Negated value
        if ((current_index & utils<sa_index>::NEGATIVE_MASK) > 0) {
            // End of unsorted interval found
            interval_end = pos;
            // Skip interval of sorted elements (negated length contained in
            // current_index)
            pos += (current_index ^ utils<sa_index>::NEGATIVE_MASK);
        } else {
            if(unsorted == 0 && pos > 0) {
            interval_begin = pos-1;
            }
            unsorted = true;
        }
        // if unsorted interval contains more than one element (after
        // interval_end has been set)
        if (unsorted > 0 && interval_end > interval_begin) {
            util::sort::introsort<sa_index, compare_suffix_ranks<sa_index>>(
                rel_ind.slice(interval_begin, interval_end), cmp);
            // Reset indicator
            unsorted = false;
            // Refresh ranks for complete isa
            recompute_isa_ltr(rel_ind, isa, cmp);
        }
    }
}

template <typename sa_index>
inline static void sort_rms_suffixes(rms_suffixes<sa_index>& rms_suf) {
    size_t pos;
    compare_suffix_ranks<sa_index> cmp(rms_suf.partial_isa, 0);
    bool unsorted = true;
    util::span<sa_index> rel_ind = rms_suf.relative_indices;
    // At most that many iterations (if we have to consider last suffix (or
    // later))
    size_t max_iterations =
        rms_suf.relative_indices.size();
    
    // TODO (optimization): Check if max_iterations can be upper bounded
    //util::floor_log2(rms_suf.relative_indices.size()) + 1;
    for (size_t iter = 0; iter < max_iterations + 1; ++iter) {
        // Sort rms-suffixes
        sort_rms_suffixes_internal(rms_suf, cmp);
        
        unsorted = false;
        // Check if still unsorted:
        pos = 0;
        //sa_index ref;
        while (pos < rel_ind.size()) {
            // Negated length of sorted interval
            if ((rel_ind[pos] & utils<sa_index>::NEGATIVE_MASK) > 0) {
                // Negated index at pos
                /*ref = (rel_ind[pos] ^ utils<sa_index>::NEGATIVE_MASK);
                DCHECK_NE(ref, sa_index(0));*/
                pos += (rel_ind[pos] ^ utils<sa_index>::NEGATIVE_MASK);
            } else {
                // interval beginning with non-negated index found ->
                // unsorted interval
                unsorted = true;
                // End inner loop - we know that there are still unsorted
                // intervals
                break;
            }
        }
        // Everything has been sorted - break outer loop
        if (unsorted == 0) {
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
    size_t correct_pos;
    auto isa = rms_suf.partial_isa;

    // Skip last index because of sentinel
    for (size_t pos = rms_suf.text.size() - 1; 1 < pos; --pos) {
        correct_pos = pos - 1;
        // RMS-Suffix in text found
        if (sa_types::is_rms_type(correct_pos, types)) {

            // If predecessor of correct_pos is l-type: negate, because not
            // considered in first induce step
            if (sa_types::is_l_type(correct_pos - 1, types)) {
                out_sa[isa[--rms_count]] =
                    correct_pos ^ utils<sa_index>::NEGATIVE_MASK;
            } else {
                // Index correct_pos considered in first induce step
                out_sa[isa[--rms_count]] = correct_pos;
            }
        }
    }
    if (sa_types::is_rms_type(0, types)) {
        DCHECK_GT(rms_count, 0);

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
