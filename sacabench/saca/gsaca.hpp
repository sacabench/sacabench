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
#include <util/signed_size_type.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::gsaca {

    class gsaca {
    public:

        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const* NAME = "GSACA";
        static constexpr char const* DESCRIPTION =
            "Computes a suffix array with the algorithm gsaca by Uwe Baier.";

        /**
         * \brief Calculates a suffix array for the given text with the gsaca algorithm.
         *
         * This algorithm is described by Uwe Baier in
         * "Linear-time Suffix Sorting - A New Approach for Suffix Array Construction"
         * ("http://vesta.informatik.rwth-aachen.de/opus/volltexte/2016/6069/pdf/LIPIcs-CPM-2016-23.pdf").
         *
         * \param text The input text for which the suffix array will be constructed.
         * \param alphabet_size Number of distinct symbols in input.
         * \param out_sa Space for the resulting suffix array.
         */
        template<typename sa_index>
        inline static void construct_sa(sacabench::util::string_span text_with_sentinels,
                                        util::alphabet const& alphabet,
                                        sacabench::util::span<sa_index> out_sa) {

            // Check if text only contains sentinel.
            if (text_with_sentinels.size() == 1) {
                return;
            }

            tdc::StatPhase gsaca("Preparation");

            // Setup needed values and build initial group structure.
            gsaca_values values = gsaca_values<sa_index>();
            size_t number_of_chars = text_with_sentinels.size();
            build_initial_structures(text_with_sentinels, alphabet, out_sa, values, number_of_chars);

            //print(values, number_of_chars, "After initializing structures.");

            gsaca.split("Phase 1");

            // Process groups in descending order. A group is defined through its start and end.
            sa_index group_start_temp = 0;
            sa_index group_end_temp = 0;
            for (values.group_end = number_of_chars - static_cast<sa_index>(1);
                    values.group_end > static_cast<sa_index>(0);
                    values.group_end = group_start_temp - static_cast<sa_index>(1)) {

                // Calculate start of current group.
                sa_index last_index_of_group = out_sa[values.group_end];
                values.group_start = values.GLINK[last_index_of_group];

                // Save borders of current group temporarily to use them later again.
                group_start_temp = values.group_start;
                group_end_temp = values.group_end;

                // Mark index of start of current group in GSIZE to check it later on.
                sa_index group_start_marker = static_cast<sa_index>(0);
                values.GSIZE[values.group_start] = group_start_marker;

                // Compute the prev pointer of the indices of current group.
                compute_prev_pointer(out_sa, values, number_of_chars);

                //print(values, number_of_chars, "After Prev Pointer Calculation.");

                // Prepares ISA for phase 2 and updates group structure.
                update_group_structure(out_sa, values);

                //print(values, number_of_chars, "After Updating Group Structure.");

                // Reorders the groups.
                size_t number_of_splitted_groups = static_cast<sa_index>(0);
                reorder_suffixes(out_sa, values, number_of_splitted_groups, group_start_temp);

                //print(values, number_of_chars, "After reordering suffixes.");

                // Rearranges previous suffixes stored in other groups.
                rearrange_suffixes(out_sa, values, number_of_splitted_groups);

                //print(values, number_of_chars, "After rearranging suffixes.");

                // Prepare current group for phase 2. Sets counter to mark place for next entry.
                out_sa[group_end_temp] = group_start_temp;
            }

            //print(values, number_of_chars, "After Phase 1.");

            gsaca.split("Phase 2");

            sort_suffixes(out_sa, values, number_of_chars);
        }

    private:

        template<typename sa_index>
        inline static sa_index prev_const() {
            return std::numeric_limits<sa_index>::max();
        }

        /**
         * \brief This struct encapsulates values, which are shared between the helper functions.
         *
         * ISA is the inverse suffix array.
         * PREV contains for each index of the suffix array the index at which the prev pointer points to.
         * GLINK contains for each index of the suffix array the index of the first elment in its group.
         * GSIZE contains the size of each group at the first index of that group. The rest of the group is filled with 0.
         * group_start is the start of the group which is currently processed.
         * group_end is the end of the group which is currently processed.
         */
        template<typename sa_index>
        struct gsaca_values {
            sacabench::util::container<sa_index> ISA;
            sacabench::util::container<sa_index> PREV;
            sacabench::util::container<sa_index> GLINK;
            sacabench::util::container<sa_index> GSIZE;
            sa_index group_start = 0;
            sa_index group_end = 0;
        };

        template<typename sa_index>
        inline static void print(gsaca_values<sa_index>& values,
                                 size_t number_of_chars,
                                 std::string message) {

            std::cout << message << std::endl;

            std::cout << "  GSIZE:  ";
            for (size_t index = 0; index < number_of_chars; index++) {
                std::cout << values.GSIZE[index] << ", ";
            }
            std::cout << std::endl;

            std::cout << "  GLINK:  ";
            for (size_t index = 0; index < number_of_chars; index++) {
                std::cout << values.GLINK[index] << ", ";
            }
            std::cout << std::endl;

            std::cout << "  PREV:   ";
            for (size_t index = 0; index < number_of_chars; index++) {
                std::cout << values.PREV[index] << ", ";
            }
            std::cout << std::endl;
        }

        /**
         * \brief This function sets up the shared values ISA, GLINK and GSIZE.
         */
        template<typename sa_index>
        inline static void build_initial_structures(sacabench::util::string_span text,
                                                    util::alphabet const& alphabet,
                                                    sacabench::util::span<sa_index> out_sa,
                                                    gsaca_values<sa_index> &values,
                                                    size_t number_of_chars) {

            // Initalise ISA, GLINK, GSIZE and PREV.
            values.ISA = sacabench::util::make_container<sa_index>(number_of_chars);
            values.GLINK = sacabench::util::make_container<sa_index>(number_of_chars);
            values.GSIZE = sacabench::util::make_container<sa_index>(number_of_chars);
            values.PREV = sacabench::util::make_container<sa_index>(number_of_chars);
            for (size_t index = 0; index < number_of_chars; index++) {
                values.PREV[index] = prev_const<sa_index>();
            }

            // Setup helper lists to count occurring chars and to calculate the cumulative count of chars.
            auto chars_count = sacabench::util::make_container<size_t>(alphabet.size_with_sentinel());
            auto chars_cumulative = sacabench::util::make_container<size_t>(alphabet.size_with_sentinel());

            // Count occurences of each char in word.
            for (sacabench::util::character current_char : text) {
                ++chars_count[current_char];
            }

            // Build cumulative counts of all chars and set up GSIZE.
            size_t cumulative_count = 0;
            for (size_t index = 0; index < alphabet.size_with_sentinel(); index++) {
                if (chars_count[index] > 0) {
                    // The char at the current index occures in the word.
                    chars_cumulative[index] = cumulative_count;
                    values.GSIZE[cumulative_count] = static_cast<sa_index>(chars_count[index]);
                    cumulative_count += chars_count[index];
                }
            }

            // Set up ISA and GLINK.
            for (size_t index = number_of_chars - 1; index < number_of_chars; index--) {
                unsigned char current_char = text[index];

                // Caluclate borders of group for current char.
                size_t group_start = chars_cumulative[current_char];
                size_t group_size = --chars_count[current_char];
                size_t group_end = group_start + group_size;

                // Save borders of group for current char in GLINK and ISA.
                values.GLINK[index] = static_cast<sa_index>(group_start);
                values.ISA[index] = static_cast<sa_index>(group_end);
                out_sa[group_end] = static_cast<sa_index>(index);
            }
        }

        /**
         * \brief This function calcualtes the prev pointer.
         */
        template<typename sa_index>
        inline static void compute_prev_pointer(sacabench::util::span<sa_index> out_sa,
                                                gsaca_values<sa_index> &values,
                                                size_t number_of_chars) {

            for (auto index = values.group_end; index >= values.group_start; --index) {
                sa_index suffix = out_sa[index];
                sa_index previous_suffix = suffix - static_cast<sa_index>(1);
                for (previous_suffix = suffix - static_cast<sa_index>(1);
                        previous_suffix < static_cast<sa_index>(number_of_chars);
                        previous_suffix = values.PREV[previous_suffix]) {

                    if (values.ISA[previous_suffix] <= values.group_end) {
                        if (values.ISA[previous_suffix] >= values.group_start) {
                            values.GSIZE[values.ISA[previous_suffix]] = static_cast<sa_index>(1);
                        }
                        break;
                    }
                }
                values.PREV[suffix] = previous_suffix;
            }
        }

        /**
         * \brief Alternative way to calculate prev pointer.
         *
         * It uses the helper function compute_prev_pointer(size_t current_index, gsaca_values values).
         * Currently it does not work the correct way for all test cases.
         * For more information see page 32ff of master thesis
         * "Linear-time Suffix Sorting - A new approach for suffix array construction" by Uwe Baier.
         */
        template<typename sa_index>
        inline static void calculate_all_prev_pointer(sacabench::util::span<sa_index> out_sa,
                                                      gsaca_values<sa_index> &values) {

            for (size_t index = values.group_start; index <= values.group_end; index++) {

                size_t selected_index = out_sa[index];

                std::vector<size_t> list_of_remaining_indices;
                while (values.PREV[selected_index] == prev_const<sa_index>()) {
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
        }

        /**
         * \brief Helperfunction to calculate prev pointer with the function calculate_all_prev_pointer.
         */
        template<typename sa_index>
        inline static size_t compute_prev_pointer(size_t current_index,
                                                  gsaca_values<sa_index> &values) {

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
        inline static void update_group_structure(sacabench::util::span<sa_index> out_sa,
                                                  gsaca_values<sa_index> &values) {

            sa_index group_size = 0;

            // Loop over all indices of current group.
            for (size_t index = values.group_start; index <= values.group_end; index++) {

                // Prepare ISA for all suffixes for phase 2.
                auto current_suffix = out_sa[index];
                values.ISA[current_suffix] = values.group_end;

                // Check if the current index is marked as the start of the current group.
                sa_index group_start_marker = 0;
                if (values.GSIZE[index] == group_start_marker) {

                    // Move marked first suffixes to end of current group.
                    auto new_index = values.group_start + group_size;
                    out_sa[new_index] = current_suffix;

                    // Increase current group size by 1.
                    group_size += 1;
                }
            }

            // Order suffixes by the number of suffixes of same group are jumped by them.
            values.group_end = values.group_start + group_size;
        }

        template<typename sa_index>
        inline static void reorder_suffixes(sacabench::util::span<sa_index> out_sa,
                                            gsaca_values<sa_index> &values,
                                            size_t &number_of_splitted_groups,
                                            size_t group_start_temp) {

            // The value of group_end is decremented in some cases and in other cases group_start is incremented,
            // so at some point group_start will no longer be less than group_end.
            while (values.group_start < values.group_end) {

                auto index = values.group_end - static_cast<sa_index>(1);
                auto saved_group_end = values.group_end;

                // The value of index is decremented in some cases, so at some point index will be smaller than group_start.
                while (index >= values.group_start) {

                    // Check if the current suffix has a valid prev pointer to another index.
                    auto current_suffix = out_sa[index];
                    auto previous_element = values.PREV[current_suffix];

                    if (previous_element != prev_const<sa_index>()) {
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
                            values.PREV[previous_element] = prev_const<sa_index>();
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
                    auto number_of_suffixes_moved_to_end = saved_group_end - values.group_end;
                    values.GSIZE[values.group_end] = number_of_suffixes_moved_to_end;

                    // Increment number_of_splitted_groups to count them.
                    number_of_splitted_groups++;
                }
            }
        }

        /**
         * \brief Rearranges suffixes into the right groups.
         *
         * This function implements the end of phase 1 of the algorithm.
         * For more information see page 36 of master thesis
         * "Linear-time Suffix Sorting - A new approach for suffix array construction" by Uwe Baier.
         */
        template<typename sa_index>
        inline static void rearrange_suffixes(sacabench::util::span<sa_index> out_sa,
                                              gsaca_values<sa_index> &values,
                                              size_t number_of_splitted_groups) {

            // Process each element of the splitted groups in descending order.
            for (size_t index = number_of_splitted_groups; index > 0; index--) {

                // Recalculate group_end.
                values.group_end = values.group_start + values.GSIZE[values.group_start];

                // Update GSIZE and move suffix to back by switching it with the last element of current group.
                for (size_t index = values.group_end - static_cast<sa_index>(1); index >= values.group_start; index--) {

                    // Calculate first suffix of current group.
                    auto current_suffix = out_sa[index];
                    auto start_index = values.GLINK[current_suffix];

                    // Decrement size of current group by one.
                    values.GSIZE[start_index]--;

                    // Calculate end index from start_index and size of the current group.
                    auto end_index = start_index + values.GSIZE[start_index];

                    // Move the current_suffix to back by exchanging it with last suffix of its group.
                    auto groupstart_suffix = out_sa[end_index];
                    auto current_suffix_position = values.ISA[current_suffix];
                    out_sa[current_suffix_position] = groupstart_suffix;
                    values.ISA[groupstart_suffix] = current_suffix_position;
                    out_sa[end_index] = current_suffix;
                    values.ISA[current_suffix] = end_index;
                }

                // Update GLINK.
                for (size_t index = values.group_start; index < values.group_end; index++) {

                    // Calculate first suffix of current group.
                    sa_index current_suffix = out_sa[index];
                    sa_index start_index = values.GLINK[current_suffix];

                    // Calculate last index of group and set it as the GLINK value of current_suffix.
                    sa_index end_index = start_index + values.GSIZE[start_index];
                    values.GLINK[current_suffix] = end_index;
                }

                // Update GSIZE.
                for (size_t index = values.group_start; index < values.group_end; index++) {

                    // Calcualte first suffix of current group.
                    sa_index current_suffix = out_sa[index];
                    sa_index start_index = values.GLINK[current_suffix];

                    // Increase GSIZE of current group.
                    values.GSIZE[start_index]++;
                }

                // Update group_start for processing the next group.
                values.group_start = values.group_end;
            }
        }

        /**
         * \brief Calculates the right order of the suffixes in their groups.
         *
         * This function implements phase 2 of the algorithm.
         * For more information see page 38 of master thesis
         * "Linear-time Suffix Sorting - A new approach for suffix array construction" by Uwe Baier.
         */
        template<typename sa_index>
        inline static void sort_suffixes(sacabench::util::span<sa_index> out_sa,
                                         gsaca_values<sa_index> &values,
                                         size_t number_of_chars) {

            // Save sentinel as first entry in SA.
            out_sa[0] = static_cast<sa_index>(number_of_chars - 1);

            // Calculate the other values in SA.
            for (size_t index = 0; index < number_of_chars; index++) {

                // Get predeseccor char of the char at current index.
                sa_index index_of_predecessor_char = out_sa[index] - static_cast<sa_index>(1);
                while (index_of_predecessor_char < static_cast<sa_index>(number_of_chars)) {

                    // Calcualte suffix_rank as the number of suffixes in lower groups.
                    sa_index suffix_rank = values.ISA[index_of_predecessor_char];

                    // Use a specific value to mark already calculated values.
                    sa_index already_calculated_marker = static_cast<sa_index>(0);
                    if (suffix_rank == already_calculated_marker) {
                        // The suffix at the current index is already calculated.
                        break;
                    }

                    // Get position of start of current group.
                    sa_index start_of_group = out_sa[suffix_rank];
                    out_sa[suffix_rank] += static_cast<sa_index>(1);
                    
                    // Move suffix at front of its group.
                    out_sa[start_of_group] = index_of_predecessor_char;

                    // Mark that suffix is placed in SA already with the previously defined marker value.
                    values.ISA[index_of_predecessor_char] = already_calculated_marker;

                    // Get the next index to be proecessed.
                    index_of_predecessor_char = values.PREV[index_of_predecessor_char];
                }
            }
        }
    }; // class gsaca
}// namespace sacabench::gsaca
