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
        };

        /**
         * This function sets up the shared values ISA, GLINK and GSIZE.
         */
        template<typename sa_index>
        inline static gsaca_values build_initial_structures(sacabench::util::string_span text,
                                                            sacabench::util::span<sa_index> out_sa,
                                                            gsaca_values values,
                                                            size_t number_of_chars) {

            // Setup helper lists to count occurring chars and to calculate the cumulative count of chars.
            size_t chars_count[ UCHAR_MAX + 1 ] = {0};
            size_t chars_cumulative[ UCHAR_MAX + 1 ] = {0};

            // Count occurences of each char in word.
            for (unsigned char current_char : text) {
                ++chars_count[current_char];
            }

            // Build cumulative counts of all chars and set up GSIZE.
            size_t cumulative_count = 0;
            for (size_t index = 0; index < (UCHAR_MAX + 1); ++index) {
                if (chars_count[index] > 0) {
                    // The char at the current index occures in the word.
                    chars_cumulative[index] = cumulative_count;
                    values.GSIZE[cumulative_count] = chars_count[index];
                    cumulative_count += chars_count[index];
                }
            }

            // Set up ISA and GLINK.
            for (size_t index = number_of_chars - 1; index < number_of_chars; --index) {
                unsigned char current_char = text[index];

                // Caluclate borders of group for current char.
                size_t group_start = chars_cumulative[current_char];
                size_t group_size = --chars_count[current_char];
                size_t group_end = group_start + group_size;

                // Save borders of group for current char in GLINK and ISA.
                values.GLINK[index] = group_start;
                values.ISA[index] = group_end;
                out_sa[group_end] = index;
            }

            // Return the changed values.
            // TODO: Use a pointer to gsaca_values to reduce needed memory.
            return values;
        }

        /**
         * This function calcualtes the prev pointer.
         */
        template<typename sa_index>
        inline static gsaca_values compute_prev_pointer(sacabench::util::span<sa_index> out_sa,
                                                        gsaca_values values,
                                                        size_t number_of_chars) {

            // Calculate prev pointer for all elements of current group in descending order.
            for (size_t index = values.group_end; index >= values.group_start; --index) {

                // Calculate current suffix and predecessor of it.
                size_t current_suffix = out_sa[index];
                size_t previous_element = current_suffix - 1;

                // Loop through all indices up to the one of the last character.
                while(previous_element < number_of_chars) {

                    // End if the current group is left.
                    if (values.ISA[previous_element] <= values.group_end) {
                        break;
                    }

                    // Check the next predecessor.
                    previous_element = values.PREV[previous_element];
                }

                // Save the calculated previous element.
                values.PREV[current_suffix] = previous_element;
            }

            // Return the changed values.
            // TODO: Use a pointer to gsaca_values to reduce needed memory.
            return values;
        }

        /**
         * This is the alternative way to calculate prev pointer from the paper.
         * It uses the helper function compute_prev_pointer(size_t current_index, gsaca_values values).
         * Currently it does not work the correct way for all test cases.
         *
         * For more information see page 32ff of master thesis
         * "Linear-time Suffix Sorting - A new approach for suffix array construction" by Uwe Baier.
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

            // Return the changed values.
            // TODO: Use a pointer to gsaca_values to reduce needed memory.
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

        template<typename sa_index>
        inline static gsaca_values update_group_structure(sacabench::util::span<sa_index> out_sa,
                                                          gsaca_values values) {

            size_t group_size = 0;

            // Loop over all indices of current group.
            for (size_t index = values.group_start; index <= values.group_end; ++index) {

                // Prepare ISA for all suffixes for phase 2.
                size_t current_suffix = out_sa[index];
                values.ISA[current_suffix] = values.group_end;

                // Check if the current index is marked as the start of the current group.
                size_t group_start_marker = 0;
                if (values.GSIZE[index] == group_start_marker) {

                    // Move marked first suffixes to end of current group.
                    size_t new_index = values.group_start + group_size;
                    out_sa[new_index] = current_suffix;

                    // Increase current group size by 1.
                    group_size += 1;
                }
            }

            // Order suffixes by the number of suffixes of same group are jumped by them.
            values.group_end = values.group_start + group_size;

            // Return the changed values.
            // TODO: Use a pointer to gsaca_values to reduce needed memory.
            return values;
        }


        template<typename sa_index>
        inline static gsaca_values reorder_suffixes(sacabench::util::span<sa_index> out_sa,
                                                    gsaca_values values,
                                                    size_t* number_of_splitted_groups,
                                                    size_t number_of_chars,
                                                    size_t group_start_temp) {

            // The value of group_end is decremented in some cases and in other cases group_start is incremented,
            // so at some point group_start will no longer be less than group_end.
            while (values.group_start < values.group_end) {

                size_t index = values.group_end - 1;
                size_t saved_group_end = values.group_end;

                // The value of index is decremented in some cases, so at some point index will be smaller than group_start.
                while (index >= values.group_start) {

                    // Check if the current suffix has a valid prev pointer to another index.
                    size_t current_suffix = out_sa[index];
                    size_t previous_element = values.PREV[current_suffix];
                    if (previous_element != -1) {
                        // Case 1: There exists a valid prev pointer.

                        // Check if the previous element is in the same or a smaller group.
                        if (values.ISA[previous_element] < group_start_temp) {
                            // Case 1.1: The previous_element is in a lex. smaller group.

                            // Move the element at which the prev pointer of the current index points to to the end of the group.
                            values.group_end--;
                            out_sa[index] = out_sa[values.group_end];
                            out_sa[values.group_end] = previous_element;

                        } else {
                            // Case 1.2: The previous_element is in the same or a lex. greater group.

                            // The prev pointer of the current suffix is changed to the same as the previous element has.
                            values.PREV[current_suffix] = values.PREV[previous_element];

                            // Mark the prev pointer so it is no longer be checked in the previous comperation.
                            values.PREV[previous_element] = -1;
                        }

                        // Decrement index for next iteration of while loop.
                        index--;

                    } else {
                        // Case 2: There is no valid prev pointer.

                        // Remove entry.
                        out_sa[index] = out_sa[values.group_start];

                        // Increment the group start for next iteration of while loop.
                        values.group_start++;
                    }
                }

                // Check if suffixes were moved to the end of their group.
                if (values.group_end < saved_group_end) {

                    // Save the number of suffixes moved to the end of their group in GSIZE.
                    size_t number_of_suffixes_moved_to_end = saved_group_end - values.group_end;
                    values.GSIZE[values.group_end] = number_of_suffixes_moved_to_end;

                    // Increment number_of_splitted_groups to count them.
                    ++*number_of_splitted_groups;
                }
            }

            // Return the changed values.
            // TODO: Use a pointer to gsaca_values to reduce needed memory.
            return values;
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
                    size_t start_index = values.GLINK[previous_element];
                    start_index += --values.GSIZE[start_index];

                    //move previous to back by exchanging it with last suffix s of group
                    size_t current_suffix = out_sa[start_index];
                    size_t tmp = values.ISA[previous_element];
                    out_sa[tmp] = current_suffix;
                    values.ISA[current_suffix] = tmp;
                    out_sa[start_index] = previous_element;
                    values.ISA[previous_element] = start_index;
                }

                //set new GLINK for moved suffixes
                for (size_t index = values.group_start; index < values.group_end; ++index) {
                    size_t previous_element = out_sa[index];
                    size_t start_index = values.GLINK[previous_element];
                    start_index += values.GSIZE[start_index];
                    values.GLINK[previous_element] = start_index;
                }

                //set up GSIZE for newly created groups
                for (size_t index = values.group_start; index < values.group_end; ++index) {
                    size_t previous_element = out_sa[index];
                    size_t start_index = values.GLINK[previous_element];
                    values.GSIZE[start_index]++;
                }

                values.group_start = values.group_end;
            }

            // Return the changed values.
            // TODO: Use a pointer to gsaca_values to reduce needed memory.
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

            // Check if text is not empty.
            if (text.size() == 0) {
                return;
            }

            // Setup needed values and build initial group structure.
            gsaca_values values = gsaca_values();
            size_t number_of_chars = text.size();
            std::fill_n(values.PREV, 1000, -1);
            values = build_initial_structures(text, out_sa, values, number_of_chars);

            //// PHASE 1 ////

            // Process groups in descending order. A group is defined through its start and end.
            size_t group_start_temp = 0;
            size_t group_end_temp = 0;
            for (values.group_end = number_of_chars - 1; values.group_end > 0; values.group_end = group_start_temp - 1) {

                // Calculate start of current group.
                size_t last_index_of_group = out_sa[values.group_end];
                values.group_start = values.GLINK[last_index_of_group];

                // Save borders of current group temporarily to use them later again.
                group_start_temp = values.group_start;
                group_end_temp = values.group_end;

                // Mark index of start of current group in GSIZE to check it later on.
                size_t group_start_marker = 0;
                values.GSIZE[values.group_start] = group_start_marker;

                // Compute the prev pointer of the indices of current group.
                values = compute_prev_pointer(out_sa, values, number_of_chars);

                // Prepares ISA for phase 2 and updates group structure.
                values = update_group_structure(out_sa, values);

                // Reorders the groups.
                size_t number_of_splitted_groups = 0;
                values = reorder_suffixes(out_sa, values, &number_of_splitted_groups, number_of_chars, group_start_temp);

                // Rearranges previous suffixes stored in other groups.
                values = rearrange_suffixes(out_sa, values, number_of_splitted_groups);

                // Prepare current group for phase 2. Sets counter to mark place for next entry.
                out_sa[group_end_temp] = group_start_temp;
            }

            //// PHASE 2 ////
            sort_suffixes(out_sa, values, number_of_chars);
        }
    }; // class gsaca
} // namespace sacabench::gsaca
