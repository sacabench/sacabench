/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "container.hpp"
#include "span.hpp"
#include "bits.hpp"


namespace sacabench::util {
/**\brief Transforms inverse Suffix Array to Suffix Array
 * \param isa calculated ISA
 * \param sa memory block for resulting SA
 *
 * This method transforms the ISA to SA using a simple scan from right to left
 */
template <typename T>
void isa2sa_simple_scan(const T &isa, T &sa) {
    for (size_t i = 0; i < isa.size(); ++i) {
        sa[isa[i]] = i;
    }
}


/**\brief Transforms inverse Suffix Array to Suffix Array, using no extra space
 * \param isa calculated ISA
 *
 * This method transforms the ISA to SA by
 * cyclically replacing ranks with suffix-indices
 * Note: ranks are always < 0!
 */
template <typename T>
void isa2sa_inplace(T &isa) {
    for (size_t i = 0; i < isa.size(); i++) {
        if (isa[i] < 0) {
            ssize_t start = i;
            ssize_t suffix = i;
            auto rank = -isa[i] - 1;
            do {
                auto tmp_rank = -isa[rank] - 1;
                isa[rank] = suffix;
                suffix = rank;
                rank = tmp_rank;
            } while (rank != start);
            isa[rank] = suffix;
        }
    }
}

/*
TODO:
template <typename T>
void isa2sa_inplace(T &isa) {

    for (size_t i = 0; i < isa.size(); i++) {
        if (isa[i] < 0) {
            ssize_t start = i;
            ssize_t suffix = i;
            auto rank = -isa[i] - 1;
        do {
            auto tmp_rank = -isa[rank] - 1;
            isa[rank] = suffix;
            suffix = rank;
            rank = tmp_rank;
        } while (rank != start);
        isa[rank] = suffix;
        }
    }
}
*/

// variant for m_suf_sort
template <typename T>
void isa2sa_inplace2(span<T> isa) {
    constexpr T END = ((T)(-1)) >> 1;
    constexpr T NEG_BIT = ((T)1) << (bits_of<T> - 1);

    for (T i = 0; i < isa.size(); ++i) {
        //if entry is a rank
        if ((isa[i] & NEG_BIT) > 0) {
            T start = i;
            T suffix = i;
            auto rank = (isa[i] & END);

        do {
            auto tmp_rank = (isa[rank] & END);
            isa[rank] = suffix;
            suffix = rank;
            rank = tmp_rank;
        } while (rank != start);
        isa[rank] = suffix;
        }
    }
}

/**\brief Transforms inverse Suffix Array to Suffix Array with less space
 * than with simple scan but more scans
 * \param isa calculated ISA
 *
 * This method transforms the ISA to SA according to
 * M. Maniscalco's and S. Puglisi's "An Efficient, Versatile Approach
 * to Suffix Sorting" (2008).
 * It scans the array multiple times and uses an extra memory block
 * with size n/2
 *
 * Note: the three MSBs of each rank must be unused!
 */
template <typename T>
void isa2sa_multiscan(T &isa) {

    size_t n = isa.size();

    // shift size to tag values according to their group
    size_t shift_size = (sizeof(size_t) * 8) - 3;

    // represents empty spot
    // maybe TODO: somehow replace with nullptr
    size_t invalid_number = 1;
    invalid_number = invalid_number << (sizeof(size_t) * 8 - 1);

    // size of xA and xB
    size_t n_quarter = (n / 4) + (n % 4 > 0);

    // init xA and xB in a successive block
    // TODO replace ssize with own type
    auto x = make_container<ssize_t>((n / 2) + (n % 2 > 0));

    // init x with invalid values
    for (size_t index = 0; index < x.size(); ++index) {
        x[index] = -1;
    }

    // move first quarter to xA
    size_t k = 1;
    for (size_t index = 0; index < n; ++index) {
        if (isa[index] < n / 4) {
            x[isa[index]] = index;
            isa[index] = invalid_number;
        }
    }
    k = 2;
    while (k <= 4) {
        size_t pointer_to_other_half = 0;
        // if k is even, move elements to xB
        if (k % 2 == 0) {
            for (size_t index = 0; index < n; ++index) {
                    if (isa[index] >= (n * (k - 1) / 4) && isa[index] < (n * k / 4)) {
                        x[isa[index] - (n * (k - 1) / 4) + n_quarter] = index;
                        isa[index] = invalid_number;
                }
                // if detected empty spot, fill it with next number in xB tagged with
                // its group
                if (isa[index] == invalid_number && pointer_to_other_half < n_quarter &&x[pointer_to_other_half] != -1) {
                    isa[index] = ((k - 1) << shift_size) | x[pointer_to_other_half];
                    x[pointer_to_other_half] = -1;
                    ++pointer_to_other_half;
                }
            }
        }
        // else move elements to xA
        else {
            for (size_t index = 0; index < n; ++index) {
                if (isa[index] >= (n * (k - 1) / 4) && isa[index] < (n * k / 4)) {
                    x[isa[index] - (n * (k - 1) / 4)] = index;
                    isa[index] = invalid_number;
                }
                // if detected empty spot, fill it with next number in xB tagged with
                // its group
                if (isa[index] == invalid_number && pointer_to_other_half < n_quarter &&x[pointer_to_other_half + n_quarter] != -1) {
                    isa[index] =((k - 1) << shift_size) | x[pointer_to_other_half + n_quarter];
                    x[pointer_to_other_half + n_quarter] = -1;
                    ++pointer_to_other_half;
                }
            }
        }
        ++k;
    }
    --k;

    // place elements of xB, which are now the last quarter, at the empty end
    // of ISA
    for (size_t index = n_quarter; index < x.size(); ++index) {
        isa[index + (n * (k - 2) / 4)] = x[index];
        x[index] = -1;
    }
    size_t most_left_free = 0;
    size_t a_pointer = 0;
    size_t b_pointer = n_quarter;
    size_t last_q1 = 0;
    size_t remove_tag_mask= std::numeric_limits<size_t>::max()>>3;

    // place other elements in ISA according to their tag
    for (size_t index = 0; index < n_quarter + (n / 2); ++index) {
        size_t group = (isa[index] >> shift_size);
        switch (group) {
            // Group 1 moves to the beginning of ISA
            case 1:
                isa[most_left_free] = isa[index] & remove_tag_mask;
                isa[index] = invalid_number;
                last_q1 = most_left_free;
                while (isa[++most_left_free] != invalid_number) {
                }
                break;
            // Group 2 moves to xA
            case 2:
                x[a_pointer] = isa[index] & remove_tag_mask;
                isa[index] = invalid_number;
                ++a_pointer;
                break;
            // Group 3 moves to xB
            case 3:
                x[b_pointer] = isa[index] & remove_tag_mask;
                isa[index] = invalid_number;
                ++b_pointer;
            break;
        }
    }
    ++last_q1;
    // move content from x (Group 2 and 3) to ISA
    for (size_t index = 0; index < x.size(); ++index) {
        isa[last_q1 + index] = x[index];
    }
    }
}
/******************************************************************************/
