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

#define C_SIZE (UCHAR_MAX+1)
#define C_CUM(c) C[c << 1]        //cumulative C array
#define C_CNT(c) C[(c << 1) + 1]  //C array counting characters
#define GLINK PREV                //alias
#define GENDLINK ISA              //alias

namespace sacabench::gsaca {

    class gsaca {
    private:
        static unsigned int *ISA, *PREV;
        static void *GSIZE;
        static unsigned int i, j;
        static unsigned int group_start, group_end;
        static unsigned int suffix, previous, sr;
        static unsigned int tmp, gstarttmp, gendtmp;
        static unsigned int C[2*C_SIZE]; //counts and cumulative counts

        template<typename sa_index>
        static void build_initial_group_structure(sacabench::util::string_span text,
                                                  sacabench::util::span<sa_index> out_sa,
                                                  unsigned int number_of_chars) {
            //build initial group structure
            for (i = 0; i < C_SIZE; ++i) {
                C_CNT(i) = 0;
            }
            for (i = 0; i < number_of_chars; ++i) {
                unsigned char current_char = text[i];
                ++C_CNT(current_char); //count characters
            }
            j = 0;
            for (i = 0; i < C_SIZE; ++i) { //build cumulative counts and set up GSIZE
                if (C_CNT(i) > 0) {
                    C_CUM(i) = j;
                    gsize_set(GSIZE, j, C_CNT(i));
                    j += C_CNT(i);
                }
            }
            for (i = number_of_chars - 1; i < number_of_chars; --i) { //set up ISA, GLINK and SA
                unsigned char current_char = text[i];
                group_start = C_CUM(current_char);
                sr = group_start + --C_CNT(current_char);
                GLINK[i] = group_start;
                ISA[i] = sr;
                out_sa[sr] = i;
            }
        }

        template<typename sa_index>
        static void calc_prev_pointer(sacabench::util::span<sa_index> out_sa,
                                      unsigned int number_of_chars) {
            //compute prev - pointers and mark suffixes of own group that
            //have a prev-pointer of own group pointing to them
            for (i = group_end; i >= group_start; --i) {
                suffix = out_sa[i]; //use prev - pointers from already used groups
                for (previous = suffix - 1; previous < number_of_chars; previous = PREV[previous]) {
                    if (ISA[previous] <= group_end) {
                        if (ISA[previous] >= group_start) {
                            gsize_set(GSIZE, ISA[previous], 1); //mark ISA[previous]
                        }
                        break;
                    }
                }
                PREV[suffix] = previous;
            }
        }

        template<typename sa_index>
        static void reorder_suffixes(sacabench::util::span<sa_index> out_sa,
                                     unsigned int number_of_chars) {
            //order the suffixes according on how much suffixes of same group are jumped by them
            group_end = group_start + j; //exclusive bound by now
            j = 0;
            do {
                i = group_end - 1;
                sr = group_end;
                while (i >= group_start) {
                    suffix = out_sa[i];
                    previous = PREV[suffix];
                    if (previous < number_of_chars) {
                        if (ISA[previous] < gstarttmp) { //p is in a lex. smaller group
                            out_sa[i--] = out_sa[--group_end];
                            out_sa[group_end] = previous; //push prev to back
                        } else { //p is in same group
                            PREV[suffix] = PREV[previous];
                            PREV[previous] = number_of_chars; //clear prev pointer, is not used in phase 2
                            --i;
                        }
                    } else { //prev points to nothing
                        out_sa[i] = out_sa[group_start++]; //remove entry
                    }
                }
                //write number of suffixes written to end on stack using GSIZE
                if (group_end < sr) {
                    gsize_set(GSIZE, group_end, sr - group_end);
                    ++j; //also, count number of splitted groups
                }
            } while (group_start < group_end);
        }

        template<typename sa_index>
        static void rearragne_suffixes(sacabench::util::span<sa_index> out_sa) {
            //rearrange previous suffixes stored in other groups
            while (j--) {
                group_end = group_start + gsize_get(GSIZE, group_start);
                //decrement group count of previous group suffixes, and move them to back
                for (i = group_end - 1; i >= group_start; --i) {
                    previous = out_sa[i];
                    sr = GLINK[previous];
                    sr += gsize_dec_get(GSIZE, sr);
                    //move previous to back by exchanging it with last suffix s of group
                    suffix = out_sa[sr];
                    tmp = ISA[previous];
                    out_sa[tmp] = suffix;
                    ISA[suffix] = tmp;
                    out_sa[sr] = previous;
                    ISA[previous] = sr;
                }
                //set new GLINK for moved suffixes
                for (i = group_start; i < group_end; ++i) {
                    previous = out_sa[i];
                    sr = GLINK[previous];
                    sr += gsize_get(GSIZE, sr);
                    GLINK[previous] = sr;
                }
                //set up GSIZE for newly created groups
                for (i = group_start; i < group_end; ++i) {
                    previous = out_sa[i];
                    sr = GLINK[previous];
                    gsize_inc(GSIZE, sr);
                }
                group_start = group_end;
            }
        }

        template<typename sa_index>
        static void sort_suffixes(sacabench::util::span<sa_index> out_sa,
                                  unsigned int number_of_chars) {
            out_sa[0] = number_of_chars - 1;
            for (i = 0; i < number_of_chars; i++) {
                suffix = out_sa[i] - 1;
                while (suffix < number_of_chars) {
                    sr = GENDLINK[suffix];
                    if (sr == number_of_chars) { //suffix already placed to SA, stop
                        break;
                    }
                    sr = out_sa[sr]++; //get position where to write s
                    out_sa[sr] = suffix;
                    //mark that suffix is placed in SA already
                    GENDLINK[suffix] = number_of_chars;
                    suffix = PREV[suffix]; //process next suffix
                }
            }
        }

    public:
        template<typename sa_index>
        static void construct_sa(sacabench::util::string_span text,
                                 size_t alphabet_size,
                                 sacabench::util::span<sa_index> out_sa) {

            size_t number_of_chars = text.size();

            //set up needed structures
            ISA = (unsigned int *) malloc(number_of_chars * sizeof(unsigned int));
            PREV = (unsigned int *) malloc(number_of_chars * sizeof(unsigned int));
            GSIZE = gsize_calloc(number_of_chars);

            //// PHASE 1: pre-sort suffixes ////
            build_initial_group_structure(text, out_sa, number_of_chars);

            //process groups from highest to lowest
            for (group_end = number_of_chars - 1; group_end > 0; group_end = gstarttmp - 1) {
                group_start = GLINK[out_sa[group_end]];
                gstarttmp = group_start;
                gendtmp = group_end;

                //clear GSIZE group size for marking
                gsize_clear(GSIZE, group_start);

                calc_prev_pointer(out_sa, number_of_chars);

                //set GENDLINK of all suffixes for phase 2 and move unmarked suffixes to the front of the actual group
                j = 0;
                for (i = group_start; i <= group_end; ++i) {
                    suffix = out_sa[i];
                    GENDLINK[suffix] = group_end;
                    if (gsize_get(GSIZE, i) == 0) { //i is not marked
                        out_sa[group_start + (j++)] = suffix;
                    }
                }

                reorder_suffixes(out_sa, number_of_chars);
                rearragne_suffixes(out_sa);

                //prepare current group for phase 2
                out_sa[gendtmp] = gstarttmp; //counter where to place next entry
            }

            //// PHASE 2: sort suffixes finally ////
            sort_suffixes(out_sa, number_of_chars);

            free(ISA);
            free(PREV);
            gsize_free(GSIZE);
        }
    }; // class gsaca
} // namespace sacabench::gsaca
