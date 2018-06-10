/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/ISAtoSA.hpp>
#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/container.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::qsufsort {

// Compare function for inital sorting
struct compare_first_character {
public:
    compare_first_character(const util::string_span _input_text)
        : input_text(_input_text) {}
    template <typename sa_index>
    bool operator()(const sa_index& a, const sa_index& b) const {
        return (input_text[a] < input_text[b]);
    }
    const util::string_span input_text;
};

template <typename sa_index>
struct compare_ranks {
public:
    compare_ranks(util::container<sa_index>& _V, size_t& _h) : V(_V), h(_h) {}

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
    const util::container<sa_index>& V;
    const size_t& h;
};

// for naive case, V is not a reference, so changes doesnt effect the comparing
// function inside one iteration
template <typename sa_index>
struct compare_ranks_naive {
public:
    compare_ranks_naive(util::container<sa_index>& _V, size_t& _h)
        : V(_V.make_copy()), h(_h) {}

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
    const util::container<sa_index> V;
    const size_t h;
};

template <typename sa_index>
class qsufsort_sub {
public:
    constexpr static sa_index NEGATIVE_MASK = sa_index(1)
                                              << (sizeof(sa_index) * 8 - 1);
    constexpr static sa_index REMOVE_NEGATIVE_MASK =
        std::numeric_limits<sa_index>::max() >> 1;

    // for trouble shooting
    template <typename T>
    static void print_array(T& arr) {
        std::cout << "SA: ";
        for (size_t i = 0; i < arr.size(); i++) {
            std::cout << (bool(arr[i] & NEGATIVE_MASK) ? "-" : "")
                      << ssize_t(arr[i] & REMOVE_NEGATIVE_MASK) << ", ";
        }
        std::cout << std::endl;
    }
    // for trouble shooting
    template <typename T, typename S>
    static void print_isa(T& arr, S& out) {
        std::cout << "V: ";
        for (size_t i = 0; i < arr.size(); i++) {
            std::cout << (bool(out[i] & NEGATIVE_MASK) ? i : arr[out[i]])
                      << ", ";
        }
        std::cout << std::endl;
    }

    // template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {

        size_t n = text.size();
        // check if n is too big
        DCHECK(util::assert_text_length<sa_index>(text.size(), 1));
        // catch trivial cases
        if (n < 2)
            return;

        // init additional arrays
        auto isa = util::make_container<sa_index>(n);
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
        init_isa(text, out_sa, isa, h);
        // since we sorted accoring to first letter, increment h
        ++h;
        // comparing function, which compares the (i+h)-th ranks
        auto compare_function = compare_ranks<sa_index>(isa, h);

        while (!is_sorted) {

            size_t counter = 0;
            // jump through array with group sizes
            while (counter < out_sa.size()) {
                // Sorted Group, check if negative bit is set
                if (bool(out_sa[counter] & NEGATIVE_MASK)) {
                    // Skip sorted group
                    counter += out_sa[counter] & REMOVE_NEGATIVE_MASK;
                }
                // unsorted group
                else {
                    // size of group changes while updating, save for jumping to
                    // right place
                    size_t tmp = isa[out_sa[counter]];
                    // sort and update unsorted group
                    sort_and_update_group(out_sa, isa, compare_function,
                                          sa_index(counter),
                                          isa[out_sa[counter]]);
                    // jump over updates group
                    counter = tmp + 1;
                }
            }
            // update group sizes
            update_group_length(out_sa, isa);
            // prefix doubling
            h = h * 2;
            is_sorted = ((out_sa[0] & REMOVE_NEGATIVE_MASK) == n);
        }
        // transform isa to sa
        util::isa2sa_simple_scan(util::span<sa_index>(isa), out_sa);
    } // construct_sa

private:
    // template <typename sa_index>
    static void init_isa(util::string_span text, util::span<sa_index> out_sa,
                         util::container<sa_index>& isa, size_t h) {
        size_t n = out_sa.size();
        // rank of last element in out_sa is always n-1
        isa[(out_sa[n - 1])] = n - 1;
        for (size_t i = n - 2; i < n; --i) {
            // Calculate V
            // if same letter-> same group
            if (text[out_sa[i + 1 + h]] == text[out_sa[i + h]]) {

                isa[out_sa[i]] = isa[out_sa[i + 1]];
            } else {
                isa[out_sa[i]] = i;
            }
        }
        // Maybe directly in loop...
        update_group_length(out_sa, isa);
    }

    // template <typename sa_index>
    static void update_group_length(util::span<sa_index> out_sa,
                                    util::container<sa_index>& isa) {
        size_t n = out_sa.size();
        size_t sorted_counter = 0;
        bool sorted_group_started = false;
        size_t dif = 0;

        for (size_t i = n - 2; i < n; --i) {

            // check if number in out_sa is negative
            // if negative the group number is simply the index
            dif = (bool(out_sa[i + 1] & NEGATIVE_MASK) ? i + 1
                                                       : isa[out_sa[i + 1]]) -
                  (bool(out_sa[i] & NEGATIVE_MASK) ? i : isa[out_sa[i]]);

            // if difference between neighbours is 1, they are sorted elements
            if (dif == 1) {
                ++sorted_counter;
                sorted_group_started = true;

            } else if (sorted_group_started) {
                out_sa[i + 2] = NEGATIVE_MASK | sorted_counter;
                sorted_counter = 0;
                sorted_group_started = false;
            }
        }
        // sentinel
        out_sa[0] = NEGATIVE_MASK | (++sorted_counter);
    }

    // template <typename sa_index, typename key_func>
    template <typename key_func>
    static void sort_and_update_group(util::span<sa_index> full_array,
                                      util::container<sa_index>& isa,
                                      key_func& cmp, sa_index start,
                                      sa_index end) {
        auto out_sa = full_array.slice(start, end + 1);
        size_t n = out_sa.size();
        const auto equal = util::as_equal(cmp);
        // recursion termination
        if (n <= 1) {
            // if only one element is in partition, set its rank
            if (n == 1)
                isa[full_array[start]] = start;
            return;
        }

        if (n == 2) {
            if (cmp(out_sa[1], out_sa[0])) {
                std::swap(out_sa[0], out_sa[1]);
            }
            if (equal(out_sa[0], out_sa[1])) {
                update_equal_partition_ranks(full_array, isa, start, end);
            }
            // if elements are not equal, it means that they are sorted, so they
            // get different ranks
            else {
                isa[full_array[start]] = start;
                isa[full_array[end]] = end;
            }
            return;
        }

        // Choose pivot according to array size
        const sa_index pivot =
            (n > util::sort::ternary_quicksort::MEDIAN_OF_NINE_THRESHOLD)
                ? util::sort::ternary_quicksort::median_of_nine(out_sa, cmp)
                : util::sort::ternary_quicksort::median_of_three(out_sa, cmp);
        auto result =
            util::sort::ternary_quicksort::partition(out_sa, cmp, pivot);
        // sorts less partition recursivly
        sort_and_update_group(full_array, isa, cmp, start,
                              sa_index(result.first + start - 1));
        // update group ranks of equal partition
        update_equal_partition_ranks(full_array, isa,
                                     sa_index(result.first + start),
                                     sa_index(result.second + start - 1));
        // sorts greater partition recursivly
        sort_and_update_group(full_array, isa, cmp,
                              sa_index(result.second + start),
                              sa_index(n - 1 + start));
        return;
    }

    // template <typename sa_index>
    static void update_equal_partition_ranks(util::span<sa_index> out_sa,
                                             util::container<sa_index>& isa,
                                             sa_index start, sa_index end) {

        DCHECK_MSG(start <= end, "Start index is bigger than end index!");
        // in an unsorted group, every elements rank is the highest index, this
        // group occuoies in the SA
        for (size_t index = start; index <= end; ++index) {

            if (bool(out_sa[index] & NEGATIVE_MASK)) {
                isa[index] = end;
            } else {
                isa[out_sa[index]] = end;
            }
        }
    }

}; // class qsufsort_sub
class qsufsort {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        qsufsort_sub<sa_index>::construct_sa(text, alphabet, out_sa);
    }

}; // class qsufsort

// keep naive version for comparison
class qsufsort_naive {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {

        size_t n = text.size();

        // catch trivial cases
        if (n < 2)
            return;

        // init additional arrays
        auto V = util::make_container<sa_index>(n);
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
            auto compare_function = compare_ranks_naive(V, h);

            sa_index counter = 0;
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
                                       counter + sa_index(L[counter]));
                    // jump over updates group
                    counter += L[counter];
                }
            }
            // finally update group sizes
            update_L(out_sa, V, L);
            // prefix doubling
            h = h * 2;
            is_sorted = (size_t(-L[0]) == n);
        }
    } // construct_sa
private:
    template <typename sa_index, typename neg_sa_index>
    static void
    init_additional_arrays(util::string_span text, util::span<sa_index> out_sa,
                           util::container<sa_index>& V,
                           util::container<neg_sa_index>& L, size_t h) {
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
    static void update_group_ranks(util::span<sa_index> out_sa,
                                   util::container<sa_index>& V, key_func& cmp,
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

    template <typename sa_index>
    static void update_L(util::span<sa_index> out_sa,
                         util::container<sa_index>& V,
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

}; // class qsufsort-naive

} // namespace sacabench::qsufsort
