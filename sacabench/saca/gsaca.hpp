/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <limits.h>
#include <stdlib.h>
#include <util/span.hpp>
#include <util/string.hpp>

#define GENDLINK gsaca_values::ISA              //alias

namespace sacabench::gsaca {

    class gsaca {
    private:

    public:
        struct gsaca_values {
            size_t ISA[1000] = {0};
            int PREV[1000] = {-1};
            size_t GLINK[1000] = {0};
            size_t GSIZE[1000] = {0};
            size_t group_start = 0;
            size_t group_end = 0;
            size_t suffix = 0;
            size_t sr = 0;
            size_t gstarttmp = 0;
            size_t gendtmp = 0;
        };

        template<typename sa_index>
        inline static gsaca_values build_initial_structures(sacabench::util::string_span text,
                                                            sacabench::util::span<sa_index> out_sa,
                                                            gsaca_values values,
                                                            size_t number_of_chars) {

            size_t chars_count[ UCHAR_MAX + 1 ] = {0};
            size_t chars_cumulative[ UCHAR_MAX + 1 ] = {0};

            // increases count of occurence of each char in word
            for (unsigned char current_char : text) {
                ++chars_count[current_char];
            }

            size_t cumulative_count = 0;
            // build cumulative counts and set up GSIZE
            for (size_t index = 0; index < (UCHAR_MAX + 1); ++index) {
                if (chars_count[index] > 0) {
                    chars_cumulative[index] = cumulative_count;
                    values.GSIZE[cumulative_count] = chars_count[index];
                    // set j to the cumulative count of the numbers up to index i
                    cumulative_count += chars_count[index];
                }
            }
            for (size_t index = number_of_chars - 1; index < number_of_chars; --index) {
                //set up values.ISA, GLINK and SA
                unsigned char current_char = text[index];
                values.group_start = chars_cumulative[current_char];
                values.sr = values.group_start + --chars_count[current_char];
                values.GLINK[index] = values.group_start;
                values.ISA[index] = values.sr;
                out_sa[values.sr] = index;
            }
            return values;
        }

        template<typename sa_index>
        inline static gsaca_values compute_prev_pointer(sacabench::util::span<sa_index> out_sa,
                                                        gsaca_values values,
                                                        size_t number_of_chars) {

            for (size_t index = values.group_end; index >= values.group_start; --index) {
                values.suffix = out_sa[index]; //use prev - pointers from already used groups
                size_t previous_element;
                for (previous_element = values.suffix - 1; previous_element < number_of_chars; previous_element = values.PREV[previous_element]) {
                    if (values.ISA[previous_element] <= values.group_end) {
                        if (values.ISA[previous_element] >= values.group_start) {
                            values.GSIZE[values.ISA[previous_element]] = 1; //mark values.ISA[previous_element]
                        }
                        break;
                    }
                }
                values.PREV[values.suffix] = previous_element;
            }
            return values;
        }

        /**
         * This is the alternative way to calculate prev pointer from the paper.
         * It uses the helper function compute_prev_pointer(size_t current_index, gsaca_values values).
         * Currently it does not work the correct way for all test cases.
         */
        template<typename sa_index>
        inline static gsaca_values calculate_all_prev_pointer(sacabench::util::span<sa_index> out_sa, gsaca_values values) {

            for (size_t index = values.group_start; index <= values.group_end; index++) {

                size_t selected_index = out_sa[index];

                std::vector<size_t> list_of_remaining_indices;
                while (values.PREV[selected_index] == -1) {
                    size_t prev = compute_prev_pointer(selected_index, values);
                    if (prev == 0 || values.ISA[prev] < values.group_start) {
                        values.PREV[selected_index] = prev;
                    } else {
                        list_of_remaining_indices.push_back(selected_index);
                        selected_index = prev;
                    }
                }
                for (size_t remaining_index : list_of_remaining_indices) {
                    values.PREV[remaining_index] = values.PREV[selected_index];
                }
            }

            return values;
        }

        inline static size_t compute_prev_pointer(size_t current_index, gsaca_values values) {
            size_t previous_index = current_index - 1;
            while (previous_index > 0) {
                if (values.ISA[previous_index] <= values.group_end) {
                    return previous_index;
                }
                previous_index = values.PREV[previous_index];
            }
            return 0;
        }

        /**
        * This function implements the end of phase 1 of the algorithm.
        *
        * For more information see page 36 of master thesis
        * "Linear-time Suffix Sorting - A new approach for suffix array construction" by Uwe Baier.
        */
        template<typename sa_index>
        inline static gsaca_values rearrange_suffixes(sacabench::util::span<sa_index> out_sa,
                                                      gsaca_values values,
                                                      size_t number_of_splitted_groups) {

            // Process each element of the splitted groups in descending order.
            for (size_t index = number_of_splitted_groups; index > 0; index --) {

                values.group_end = values.group_start + values.GSIZE[values.group_start];

                //decrement group count of previous group suffixes, and move them to back
                for (size_t index = values.group_end - 1; index >= values.group_start; --index) {

                    // calucalte last suffix of current group
                    size_t previous_element = out_sa[index];
                    values.sr = values.GLINK[previous_element];
                    values.sr += --values.GSIZE[values.sr];

                    //move previous to back by exchanging it with last suffix s of group
                    values.suffix = out_sa[values.sr];
                    size_t tmp = values.ISA[previous_element];
                    out_sa[tmp] = values.suffix;
                    values.ISA[values.suffix] = tmp;
                    out_sa[values.sr] = previous_element;
                    values.ISA[previous_element] = values.sr;
                }

                //set new GLINK for moved suffixes
                for (size_t index = values.group_start; index < values.group_end; ++index) {
                    size_t previous_element = out_sa[index];
                    values.sr = values.GLINK[previous_element];
                    values.sr += values.GSIZE[values.sr];
                    values.GLINK[previous_element] = values.sr;
                }

                //set up GSIZE for newly created groups
                for (size_t index = values.group_start; index < values.group_end; ++index) {
                    size_t previous_element = out_sa[index];
                    values.sr = values.GLINK[previous_element];
                    values.GSIZE[values.sr]++;
                }

                values.group_start = values.group_end;
            }
            return values;
        }

        /**
        * This function implements phase 2 of the algorithm.
        *
        * For more information see page 38 of master thesis
        * "Linear-time Suffix Sorting - A new approach for suffix array construction" by Uwe Baier.
        */
        template<typename sa_index>
        inline static void sort_suffixes(sacabench::util::span<sa_index> out_sa,
                                         gsaca_values values,
                                         size_t number_of_chars) {

            // Save sentinel as first entry in SA.
            out_sa[0] = number_of_chars - 1;

            // Calculate the other values in SA.
            for (size_t index = 0; index < number_of_chars; index++) {

                // Get predeseccor char of the char at current index.
                size_t index_of_predecessor_char = out_sa[index] - 1;
                while (index_of_predecessor_char < number_of_chars) {

                    // Calcualte suffix_rank as the number of suffixes in lower groups.
                    size_t suffix_rank = values.ISA[index_of_predecessor_char];

                    // Use a specific value to mark already calculated values.
                    size_t already_calculated_marker = 0;
                    if (suffix_rank == already_calculated_marker) {
                        // The suffix at the current index is already calculated.
                        break;
                    }

                    // Get position of start of current group.
                    size_t start_of_group = out_sa[suffix_rank]++;
                    // Move suffix at front of its group.
                    out_sa[start_of_group] = index_of_predecessor_char;

                    // Mark that suffix is placed in SA already with the previously defined marker value.
                    values.ISA[index_of_predecessor_char] = already_calculated_marker;

                    // Get the next index to be proecessed.
                    index_of_predecessor_char = values.PREV[index_of_predecessor_char];
                }
            }
        }

        template<typename sa_index>
        inline static void construct_sa(sacabench::util::string_span text,
                                        size_t alphabet_size,
                                        sacabench::util::span<sa_index> out_sa) {

            gsaca_values values = gsaca_values();

            if (text.size() == 0) {
                return;
            }

            size_t number_of_chars = text.size();

            //set up needed structures
            std::fill_n(values.PREV, 1000, -1);

            //// PHASE 1: pre-sort suffixes ////
            //build initial group structure
            values = build_initial_structures(text, out_sa, values, number_of_chars);

            //process groups from highest to lowest
            for (values.group_end = number_of_chars - 1; values.group_end > 0; values.group_end = values.gstarttmp - 1) {
                values.group_start = values.GLINK[ out_sa[values.group_end] ];
                values.gstarttmp = values.group_start;
                values.gendtmp = values.group_end;

                //clear GSIZE group size for marking
                values.GSIZE[values.group_start] = 0;

                //compute prev - pointers and mark suffixes of own group that
                //have a prev-pointer of own group pointing to them

                //There are two possible implementations for calculating the prev pointers.
                //The first one is the reference implementation von Uwe Baier.
                //The second one is the implementation from the paper. It fails for some examples.
                //Choose one by selecting it here.

                values = compute_prev_pointer(out_sa, values, number_of_chars);
                //values = calculate_all_prev_pointer(out_sa, values);

                //set GENDLINK of all suffixes for phase 2 and move unmarked suffixes to the front of the actual group
                size_t group_size = 0;
                for (size_t index = values.group_start; index <= values.group_end; ++index) {
                    values.suffix = out_sa[index];
                    values.GENDLINK[values.suffix] = values.group_end;
                    if (values.GSIZE[index] == 0) { //index is not marked
                        out_sa[values.group_start+(group_size++)] = values.suffix;
                    }
                }

                //order the suffixes according on how much suffixes of same group are jumped by them
                values.group_end = values.group_start + group_size; //exclusive bound by now

                size_t number_of_splitted_groups = 0;

                do {
                    size_t index = values.group_end - 1;
                    values.sr = values.group_end;
                    while (index >= values.group_start) {
                        values.suffix = out_sa[index];
                        size_t previous_element = values.PREV[values.suffix];
                        if (previous_element < number_of_chars) {
                            if (values.ISA[previous_element] < values.gstarttmp) { //p is in a lex. smaller group
                                out_sa[index--] = out_sa[--values.group_end];
                                out_sa[values.group_end] = previous_element; //push prev to back
                            } else { //p is in same group
                                values.PREV[values.suffix] = values.PREV[previous_element];
                                values.PREV[previous_element] = number_of_chars; //clear prev pointer, is not used in phase 2
                                --index;
                            }
                        } else { //prev points to nothing
                            out_sa[index] = out_sa[values.group_start++]; //remove entry
                        }
                    }
                    //write number of suffixes written to end on stack using GSIZE
                    if (values.group_end < values.sr) {
                        values.GSIZE[values.group_end] = values.sr - values.group_end;
                        ++number_of_splitted_groups; //also, count number of splitted groups
                    }
                } while (values.group_start < values.group_end);

                //rearrange previous suffixes stored in other groups
                values = rearrange_suffixes(out_sa, values, number_of_splitted_groups);

                //prepare current group for phase 2
                out_sa[values.gendtmp] = values.gstarttmp; //counter where to place next entry
            }

            //// PHASE 2: sort suffixes finally ////
            sort_suffixes(out_sa, values, number_of_chars);
        }
    }; // class gsaca
} // namespace sacabench::gsaca
