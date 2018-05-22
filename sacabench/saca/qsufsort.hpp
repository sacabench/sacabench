/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::qsufsort {

// Compare function for inital sorting
struct compare_first_character {
public:
    compare_first_character(const util::string_span& _input_text)
        : input_text(_input_text) {}
    template <typename sa_index>
    bool operator()(const sa_index& a, const sa_index& b) const {
        return ((input_text[a] - input_text[b]) < 0);
    }
    const util::string_span& input_text;
};

struct compare_ranks {
public:
    template <typename sa_index>
    compare_ranks(util::container<sa_index>& _V, size_t& _h) : V(_V), h(_h) {}

    template <typename sa_index>
    // Warum const?
    bool operator()(const sa_index& a, const sa_index& b) const {
        bool a_out_of_bound = a + h >= V.size();
        bool b_out_of_bound = b + h >= V.size();
        if (a_out_of_bound && b_out_of_bound) {
            return false;
        } else if (a_out_of_bound) {
            return true;
        } else if (b_out_of_bound) {
            return false;
        }

        return (V[a + h] < V[b + h]);
    }
    const util::container<size_t> V;
    const size_t h;
};

class qsufsort {
public:
    template <typename sa_index>
    static void construct_sa(util::string_span text, size_t& alphabet_size,
                             util::span<sa_index> out_sa) {

        size_t n = text.size();

        // catch trivial cases
        if (n < 2)
            return;

        // init additional arrays
        auto V = util::make_container<size_t>(n);
        auto L = util::make_container<ssize_t>(n);

        // init out_sa (necessary?)
        for (size_t i = 0; i < n; ++i) {
            out_sa[i] = i;
        }
        // init h (checked prefix length)
        size_t h = 0;
        // for are more readible while condition
        bool is_sorted = false;
        // comparing function for inital sort according to first character
        auto compare_first_char_function = compare_first_character(text);
        // Sort according to first character
        util::sort::ternary_quicksort::ternary_quicksort(
            out_sa, compare_first_char_function);
        // Inital calculation of V and L
        init_additional_arrays(text, out_sa, V, L, h);
        // since we sorted accoring to first letter, increment h
        ++h;
        while (!is_sorted) {

            // comparing function, which compares the (i+h)-th ranks
            auto compare_function = compare_ranks(V, h);

            size_t counter = 0;
            // jump through array with group sizes
            while (counter < out_sa.size()) {

                // Sorted Group
                if (L[counter] < 0) {
                    // Skip sorted group
                    counter -= L[counter];
                }
                // unsorted group
                else {
                    // sort unsorted group
                    util::sort::ternary_quicksort::ternary_quicksort(
                        out_sa.slice(counter, counter + L[counter]),
                        compare_function);
                    // update ranks within group
                    update_group_ranks(out_sa, V, compare_function, counter,
                                       counter + L[counter]);
                    // jump over updates group
                    counter += L[counter];
                }
            }
            // finally update group sizes
            update_L(out_sa, V, L);
            // prefix doubling
            h = h * 2;

            // size_t for supressing warning
            is_sorted = (size_t(-L[0]) == n);
        }
    } // construct_sa
private:
    template <typename sa_index>
    static void init_additional_arrays(util::string_span text,
                                       util::span<sa_index> out_sa,
                                       util::container<size_t>& V,
                                       util::container<ssize_t>& L, size_t h) {
        size_t n = out_sa.size();
        // TODO Remove if use sentinal
        size_t unsorted_counter = 0;
        size_t sorted_counter = 0;
        bool sorted_group_started = false;
        size_t dif = 0;
        // rank of last element in out_sa is always n-1
        V[out_sa[n - 1]] = n - 1;
        for (size_t i = n - 2; i < n; --i) {
            // Calculate V
            // if same letter-> same group
            if (text[out_sa[i + 1 + h]] == text[out_sa[i + h]]) {

                V[out_sa[i]] = V[out_sa[i + 1]];
            } else {
                V[out_sa[i]] = i;
            }

            // Calculate L

            // difference of ranks of adjacent suffixes
            dif = V[out_sa[i + 1]] - V[out_sa[i]];

            // count for last position..
            unsorted_counter = (dif == 0) ? unsorted_counter + 1 : 0;

            // in sorted group
            if (dif == 1) {
                ++sorted_counter;
                sorted_group_started = true;
            }
            // when unsorted group begins
            else {
                if (sorted_group_started) {
                    L[i + 2] = -sorted_counter;
                    sorted_counter = 0;
                    sorted_group_started = false;
                } else {
                    L[i + 1] = V[out_sa[i + 1]] - V[out_sa[i]];
                }
            }
        }

        // easier if use sentinal...
        if (V[out_sa[0]] == V[out_sa[1]]) {
            L[0] = ++unsorted_counter;
        } else {
            L[0] = -(++sorted_counter);
        }
    }
    template <typename sa_index, typename key_func>
    static void update_group_ranks(util::span<sa_index>& out_sa,
                                   util::container<size_t>& V, key_func& cmp,
                                   sa_index start, sa_index end) {

        const auto less = cmp;
        const auto equal = util::as_equal(cmp);
        // save highest group number of giver group
        auto group_number = V[out_sa[end - 1]];
        // for counting elements, which are still equal
        size_t to_decrease = 1;
        for (size_t index = end - 2; index >= start && index < end; --index) {
            // if equal, increase counter
            if (equal(out_sa[index], out_sa[index + 1])) {
                ++to_decrease;
            } else if (less(out_sa[index], out_sa[index + 1])) {
                // if actually less, decrease by number of seen equal elements
                // right behind
                group_number -= to_decrease;
                to_decrease = 1;
            }
            // set group number
            V[out_sa[index]] = group_number;
        }
    }

    // To be removed with improvements
    template <typename sa_index>
    static void update_L(util::span<sa_index>& out_sa,
                         util::container<size_t>& V,
                         util::container<ssize_t>& L) {

        size_t n = out_sa.size();

        size_t unsorted_counter = 0;
        size_t sorted_counter = 0;
        bool sorted_group_started = false;
        size_t dif = 0;
        for (size_t i = n - 2; i < n; --i) {

            // Calculate L
            dif = V[out_sa[i + 1]] - V[out_sa[i]];

            // count for last position..
            unsorted_counter = (dif == 0) ? unsorted_counter + 1 : 0;

            if (dif == 1) {
                ++sorted_counter;
                sorted_group_started = true;
                /*
                //not neccessary, just for testing
                L[i+1]=0;*/
            } else {
                if (sorted_group_started) {
                    L[i + 2] = -sorted_counter;
                    sorted_counter = 0;
                    sorted_group_started = false;
                } else {
                    L[i + 1] = V[out_sa[i + 1]] - V[out_sa[i]];
                }
            }
        }
        // easier if use sentinal...
        if (V[out_sa[0]] == V[out_sa[1]]) {
            L[0] = ++unsorted_counter;
        } else {
            L[0] = -(++sorted_counter);
        }
    }

}; // class qsufsort

} // namespace sacabench::qsufsort
