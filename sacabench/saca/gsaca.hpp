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
#include "gsaca_helper.hpp"

#define GLINK gsaca_values::PREV                //alias
#define GENDLINK gsaca_values::ISA              //alias

namespace sacabench::gsaca {

    class gsaca {
    private:
        struct gsaca_values {
            size_t *ISA = 0;
            size_t *PREV = 0;
            void *GSIZE;
            size_t i = 0;
            size_t group_start = 0;
            size_t group_end = 0;
            size_t suffix = 0;
            size_t previous = 0;
            size_t sr = 0;
            size_t tmp = 0;
            size_t gstarttmp = 0;
            size_t gendtmp = 0;
        };

        template<typename sa_index>
        static void build_initial_structures(sacabench::util::string_span text,
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
            for (values.i = 0; values.i < (UCHAR_MAX + 1); ++values.i) {
                if (chars_count[values.i] > 0) {
                    chars_cumulative[values.i] = cumulative_count;
                    gsize_set(values.GSIZE, cumulative_count, chars_count[values.i]);
                    // set j to the cumulative count of the numbers up to index i
                    cumulative_count += chars_count[values.i];
                }
            }
            for (values.i = number_of_chars - 1; values.i < number_of_chars; --values.i) { //set up values.ISA, GLINK and SA
                unsigned char current_char = text[values.i];
                values.group_start = chars_cumulative[current_char];
                values.sr = values.group_start + --chars_count[current_char];
                values.GLINK[values.i] = values.group_start;
                values.ISA[values.i] = values.sr;
                out_sa[values.sr] = values.i;
            }
        }

        template<typename sa_index>
        static void compute_prev_pointer(sacabench::util::span<sa_index> out_sa,
                                         gsaca_values values,
                                         size_t number_of_chars) {

            for (values.i = values.group_end; values.i >= values.group_start; --values.i) {
                values.suffix = out_sa[values.i]; //use prev - pointers from already used groups
                for (values.previous = values.suffix - 1; values.previous < number_of_chars; values.previous = values.PREV[values.previous]) {
                    if (values.ISA[values.previous] <= values.group_end) {
                        if (values.ISA[values.previous] >= values.group_start) {
                            gsize_set(values.GSIZE, values.ISA[values.previous], 1); //mark values.ISA[values.previous]
                        }
                        break;
                    }
                }
                values.PREV[values.suffix] = values.previous;
            }
        }

        template<typename sa_index>
        static void reorder_suffixes(sacabench::util::span<sa_index> out_sa,
                                     gsaca_values values,
                                     size_t number_of_chars) {

            // TOOD
        }

        template<typename sa_index>
        static void rearrange_suffixes(sacabench::util::span<sa_index> out_sa,
                                       gsaca_values values,
                                       size_t number_of_splitted_groups) {

            while (number_of_splitted_groups--) {
                values.group_end = values.group_start + gsize_get(values.GSIZE, values.group_start);
                //decrement group count of previous group suffixes, and move them to back
                for (values.i = values.group_end - 1; values.i >= values.group_start; --values.i) {
                    values.previous = out_sa[values.i];
                    values.sr = values.GLINK[values.previous];
                    values.sr += gsize_dec_get(values.GSIZE, values.sr);
                    //move previous to back by exchanging it with last suffix s of group
                    values.suffix = out_sa[values.sr];
                    values.tmp = values.ISA[values.previous];
                    out_sa[values.tmp] = values.suffix;
                    values.ISA[values.suffix] = values.tmp;
                    out_sa[values.sr] = values.previous;
                    values.ISA[values.previous] = values.sr;
                }
                //set new GLINK for moved suffixes
                for (values.i = values.group_start; values.i < values.group_end; ++values.i) {
                    values.previous = out_sa[values.i];
                    values.sr = values.GLINK[values.previous];
                    values.sr += gsize_get(values.GSIZE, values.sr);
                    values.GLINK[values.previous] = values.sr;
                }
                //set up GSIZE for newly created groups
                for (values.i = values.group_start; values.i < values.group_end; ++values.i) {
                    values.previous = out_sa[values.i];
                    values.sr = values.GLINK[values.previous];
                    gsize_inc(values.GSIZE, values.sr);
                }
                values.group_start = values.group_end;
            }
        }

        template<typename sa_index>
        static void sort_suffixes(sacabench::util::span<sa_index> out_sa,
                                  gsaca_values values,
                                  size_t number_of_chars) {

            out_sa[0] = number_of_chars - 1;
            for (values.i = 0; values.i < number_of_chars; values.i++) {
                values.suffix = out_sa[values.i] - 1;
                while (values.suffix < number_of_chars) {
                    values.sr = values.GENDLINK[values.suffix];
                    if (values.sr == number_of_chars) { //suffix already placed to SA, stop
                        break;
                    }
                    values.sr = out_sa[values.sr]++; //get position where to write s
                    out_sa[values.sr] = values.suffix;
                    //mark that suffix is placed in SA already
                    values.GENDLINK[values.suffix] = number_of_chars;
                    values.suffix = values.PREV[values.suffix]; //process next suffix
                }
            }
        }

    public:
        template<typename sa_index>
        static void construct_sa(sacabench::util::string_span text,
                                 size_t alphabet_size,
                                 sacabench::util::span<sa_index> out_sa) {

            gsaca_values values = gsaca_values();

            if (text.size() == 0) {
                return;
            }

            size_t number_of_chars = text.size();

            //set up needed structures
            values.ISA = (size_t *) malloc( number_of_chars * sizeof(size_t) );
            values.PREV = (size_t *) malloc( number_of_chars * sizeof(size_t) );
            values.GSIZE = gsize_calloc( number_of_chars );

            //// PHASE 1: pre-sort suffixes ////
            //build initial group structure
            build_initial_structures(text, out_sa, values, number_of_chars);

            //process groups from highest to lowest
            for (values.group_end = number_of_chars - 1; values.group_end > 0; values.group_end = values.gstarttmp - 1) {
                values.group_start = values.GLINK[ out_sa[values.group_end] ];
                values.gstarttmp = values.group_start;
                values.gendtmp = values.group_end;

                //clear GSIZE group size for marking
                gsize_clear(values.GSIZE, values.group_start);

                //compute prev - pointers and mark suffixes of own group that
                //have a prev-pointer of own group pointing to them
                compute_prev_pointer(out_sa, values, number_of_chars);

                //set GENDLINK of all suffixes for phase 2 and move unmarked suffixes to the front of the actual group
                size_t group_size = 0;
                for (values.i = values.group_start; values.i <= values.group_end; ++values.i) {
                    values.suffix = out_sa[values.i];
                    values.GENDLINK[values.suffix] = values.group_end;
                    if (gsize_get(values.GSIZE, values.i) == 0) { //i is not marked
                        out_sa[values.group_start+(group_size++)] = values.suffix;
                    }
                }

                //order the suffixes according on how much suffixes of same group are jumped by them
                values.group_end = values.group_start + group_size; //exclusive bound by now

                size_t number_of_splitted_groups = 0;
                do {
                    values.i = values.group_end - 1;
                    values.sr = values.group_end;
                    while (values.i >= values.group_start) {
                        values.suffix = out_sa[values.i];
                        values.previous = values.PREV[values.suffix];
                        if (values.previous < number_of_chars) {
                            if (values.ISA[values.previous] < values.gstarttmp) { //p is in a lex. smaller group
                                out_sa[values.i--] = out_sa[--values.group_end];
                                out_sa[values.group_end] = values.previous; //push prev to back
                            } else { //p is in same group
                                values.PREV[values.suffix] = values.PREV[values.previous];
                                values.PREV[values.previous] = number_of_chars; //clear prev pointer, is not used in phase 2
                                --values.i;
                            }
                        } else { //prev points to nothing
                            out_sa[values.i] = out_sa[values.group_start++]; //remove entry
                        }
                    }
                    //write number of suffixes written to end on stack using GSIZE
                    if (values.group_end < values.sr) {
                        gsize_set(values.GSIZE, values.group_end, values.sr - values.group_end);
                        ++number_of_splitted_groups; //also, count number of splitted groups
                    }
                } while (values.group_start < values.group_end);

                //rearrange previous suffixes stored in other groups
                rearrange_suffixes(out_sa, values, number_of_splitted_groups);

                //prepare current group for phase 2
                out_sa[values.gendtmp] = values.gstarttmp; //counter where to place next entry
            }

            //// PHASE 2: sort suffixes finally ////
            sort_suffixes(out_sa, values, number_of_chars);

            free(values.ISA);
            free(values.PREV);
            gsize_free(values.GSIZE);
        }
    }; // class gsaca
} // namespace sacabench::gsaca
