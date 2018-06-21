#pragma once

#include <cmath>
#include <iostream>
#include <util/assertions.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <utility>

namespace sacabench::div_suf_sort {

template <typename sa_index>
struct compare_rms_substrings {
public:
    inline compare_rms_substrings(
        const util::string_span text,
        util::container<std::pair<sa_index, sa_index>>& substrings)
        : input(text), substrings(substrings) {
        /*std::cout << "Initializing rms-substrings compare fct." << std::endl;*/
    }

    inline bool operator()(const sa_index& elem,
                           const sa_index& compare_to) const {
        // DCHECK_NE(elem, compare_to);
        if (elem == compare_to) {
            return false;
        }
        const bool elem_too_large = (elem >= substrings.size());
        const bool compare_to_too_large = (compare_to >= substrings.size());

        if (elem_too_large) {
            if (compare_to_too_large) {
                // Check how to handle this case (possibly the shorter one is
                // smaller)
                return elem < compare_to;
            }
            return true;
        }
        if (compare_to_too_large) {
            DCHECK_EQ(elem_too_large, false);
            return false;
        }
        sa_index elem_size =
            std::get<1>(substrings[elem]) - std::get<0>(substrings[elem]) + 1;
        sa_index compare_to_size = std::get<1>(substrings[compare_to]) -
                                   std::get<0>(substrings[compare_to]) + 1;
        sa_index max_pos = std::min(elem_size, compare_to_size);
        sa_index elem_begin = std::get<0>(substrings[elem]);
        sa_index compare_to_begin = std::get<0>(substrings[compare_to]);
        sa_index elem_index = elem_begin + 2,
                 compare_to_index = compare_to_begin + 2;

        for (sa_index pos = 2; pos <= max_pos; ++pos) {
            /*std::cout << "Current index :" << pos << std::endl;
            std::cout << "Comparing " << (size_t)input[elem_index] << " to "
                      << (size_t)input[compare_to_index] << std::endl;*/
            if (input[elem_index] == input[compare_to_index]) {
                ++elem_index;
                ++compare_to_index;
            } else {
                /*std::cout << "Symbol " << (size_t)input[elem_index]
                          << " differs from " << (size_t)input[compare_to_index]
                          << std::endl;*/
                return input[elem_index] < input[compare_to_index];
            }
        }
        /*std::cout << "Substrings have been the same until now." << std::endl;*/
        // If one substring is shorter than the other and they are the same
        // until now:
        elem_size =
            std::get<1>(substrings[elem]) - std::get<0>(substrings[elem]);
        compare_to_size = std::get<1>(substrings[compare_to]) -
                          std::get<0>(substrings[compare_to]);
        // Either they differ in length (shorter string is smaller) or they have
        // the same length (i.e. return false)
        return (elem_size == compare_to_size) ? false
                                              : elem_size < compare_to_size;
    }

private:
    const util::string_span input;
    util::container<std::pair<sa_index, sa_index>>& substrings;
};

// FIXME
template <typename sa_index>
struct compare_suffix_ranks {
    sa_index depth;

    inline compare_suffix_ranks(util::span<sa_index> partial_isa,
                                sa_index depth)
        : depth(depth), partial_isa(partial_isa) {
        /*std::cout << "Initializing suffix ranks compare fct." << std::endl;*/
    }

    inline bool operator()(const sa_index& elem,
                           const sa_index& compare_to) const {
        // Could cause overflow if depth is too big (especially for sa_index
        // type)
        const size_t elem_at_depth = elem + pow(2, depth);
        const size_t compare_to_at_depth = compare_to + pow(2, depth);
        /*std::cout << "elem: " << elem_at_depth
                  << ", compare_to: " << compare_to_at_depth << std::endl;*/
        const bool elem_too_large = elem_at_depth >= partial_isa.size();
        const bool compare_to_too_large =
            compare_to_at_depth >= partial_isa.size();

        if (elem_too_large) {
            if (compare_to_too_large) {
                // Both "out of bounds" -> bigger index means string ends
                // earlier (i.e. smaller)
                // TODO: Check if this condition always holds.
                /*std::cout << "Both indices out of bounds." << std::endl;*/
                return elem_at_depth > compare_to_at_depth;
            }
            /*std::cout << "elem out of bounds" << std::endl;*/
            // Only first suffix (substring) ends "behind" sentinel
            return true;
        } else if (compare_to_too_large) {
            /*std::cout << "compare_to out of bounds" << std::endl;*/
            // Only second suffix (substring) ends "behind" sentinel
            return false;
        }
        // Neither index "out of bounds":
        // Ranks of compared substrings decide order
        /*std::cout << "returns " << partial_isa[elem_at_depth] << " < "
                  << partial_isa[compare_to_at_depth] << ": "
                  << (partial_isa[elem_at_depth] <
                      partial_isa[compare_to_at_depth])
                  << std::endl;*/
        return partial_isa[elem_at_depth] < partial_isa[compare_to_at_depth];
    }

private:
    util::span<sa_index> partial_isa;
};

} // namespace sacabench::div_suf_sort
