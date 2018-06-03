/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>
#include <vector>

#include <util/sort/std_sort.hpp>
#include <util/string.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/signed_size_type.hpp>

namespace sacabench::saca::sais {
using namespace sacabench::util;

class Sais {
    public:
        #define chr(i) (cs == sizeof(ssize) ? ((ssize*)s)[i] : ((unsigned char*)s)[i])

        static const ssize L_Type = 0;
        static const ssize S_Type = 1;

        static void compute_types(ssize* t, ssize* p_1, unsigned char* s, ssize len, ssize cs) {
            t[len - 1] = S_Type;

            for (ssize i = len - 2; i >= 0; i--) {
                if (chr(i + 1) < chr(i)) {
                    t[i] = L_Type;
                    if (t[i + 1] == S_Type) { p_1[i + 1] = 1; }
                }
                else if (chr(i + 1) > chr(i)) { t[i] = S_Type; }
                else { t[i] = t[i + 1]; }
            }
        }

        static void generate_buckets(unsigned char* s, ssize len, ssize* buckets, ssize K, bool end, ssize cs) {
            ssize sum = 0;

            for (ssize i = 0; i <= K; i++) { buckets[i] = 0; }

            // bucket size for each char
            for (ssize i = 0; i < len; i++) { buckets[chr(i)]++; }

            // sum up to bucket ends
            for (ssize i = 0; i <= K; i++) {
                sum += buckets[i];
                buckets[i] = (end ? sum : sum - buckets[i]);
            }
        }

        static void induce_L_Types(unsigned char* s, ssize len, ssize *buckets, ssize K, bool end, ssize* SA, ssize* t, ssize cs) {
            generate_buckets(s, len, buckets, K, end, cs);
            for (ssize i = 0; i < len; i++) {
                ssize pre_index = SA[i] - 1; // pre index of the ith suffix array position

                if (pre_index >= 0 && t[pre_index] == L_Type) { // pre index is type L
                    SA[buckets[chr(pre_index)]++] = pre_index; // "sort" index in the bucket
                }
            }
        }

        static void induce_S_Types(unsigned char* s, ssize len, ssize *buckets, ssize K, bool end, ssize* SA, ssize *t, ssize cs) {
            generate_buckets(s, len, buckets, K, end, cs);
            for (ssize i = len - 1; i >= 0; i--) {
                ssize pre_index = SA[i] - 1;

                if (pre_index >= 0 && t[pre_index] == S_Type) {
                    SA[--buckets[chr(pre_index)]] = pre_index;
                }
            }
        }

        static void SAIS(unsigned char* s, ssize* SA, ssize len, ssize K, ssize cs) {
            ssize* t = new ssize[len]();
            ssize* p_1 = new ssize[len]();
            ssize *buckets = new ssize[K + 1]();

            compute_types(t, p_1, s, len, cs);

            generate_buckets(s, len, buckets, K, true, cs);
            // Initialize each entry in SA with -1
            for (ssize i = 0; i < len; i++) { SA[i] = -1; }
            // iterate from left to right (starting by 1 cause 0 can never be LMS) and put LMS to end of the bucket and move bucket's tail backwards
            for (ssize i = 1; i < len; i++) {
                if (p_1[i] == 1) {
                    SA[--buckets[chr(i)]] = i;
                }
            }

            // sort LMS substrings
            induce_L_Types(s, len, buckets, K, false, SA, t, cs);
            induce_S_Types(s, len, buckets, K, true, SA, t, cs);

            // because we have at most n/2 LMS, we can store the sorted indices in the first half of the SA
            ssize n1 = 0;
            for (ssize i = 0; i < len; i++) {
                if (p_1[SA[i]] == 1) {
                    SA[n1++] = SA[i];
                }
            }

            // All LMS are now stored in the (at most) first half of the SA, so the rest half of the suffix array can be used 
            for (ssize i = n1; i < len; i++) { SA[i] = -1; }

            // The given names correspond to the buckets the LMS are sorted ssizeo. To find the names, the strings have to be compared by its type and lexicographical value per char
            ssize name = 0;
            ssize previous_LMS = -1;
            for (ssize i = 0; i < n1; i++) { // max n/2 iterations
                bool diff = false;
                // compare types and chars
                ssize current_LMS = SA[i];
                for (ssize j = 0; j < len; j++) {
                    if (previous_LMS == -1 || chr(current_LMS + j) != chr(previous_LMS + j) || t[current_LMS + j] != t[previous_LMS + j]) {
                        diff = true;
                        break;
                    }
                    else if (j > 0 && (p_1[current_LMS + j] == 1 || p_1[previous_LMS + j] == 1)) { // check if next LMS is reached
                        break; // no diff was found
                    }
                }

                if (diff) { // if diff was found, adjust the name and continue with current LMS as previous
                    name++;
                    previous_LMS = current_LMS;
                }

                current_LMS = (current_LMS % 2 == 0) ? current_LMS / 2 : (current_LMS - 1) / 2;
                SA[n1 + current_LMS] = name - 1;
            }

            // not needed
            for (ssize i = len - 1, j = len - 1; i >= n1; i--) {
                if (SA[i] >= 0) {
                    SA[j--] = SA[i];
                }
            }

            // if there are more as less names than LMS, we have duplicates and have to recurse
            ssize* SA1 = SA;
            ssize* s1 = (SA + len - n1);
            if (name < n1) {
                // new string that consists of ssizeegers now, as they represent the names
                SAIS((unsigned char*)s1, SA1, n1, name - 1, sizeof(ssize));
            }
            else {
                for (ssize i = 0; i < n1; i++) {
                    SA1[s1[i]] = i;
                }
            }

            // induce the final SA
            generate_buckets(s, len, buckets, K, true, cs);
            ssize j;
            for (ssize i = 1, j = 0; i < len; i++) {
                if (p_1[i] == 1) {
                    s1[j++] = i;
                }
            }
            for (ssize i = 0; i < n1; i++) {
                SA1[i] = s1[SA1[i]];
            }
            for (ssize i = n1; i < len; i++) {
                SA[i] = -1;
            }
            for (ssize i = n1 - 1; i >= 0; i--) {
                j = SA[i];
                SA[i] = -1;
                SA[--buckets[chr(j)]] = j;
            }

            induce_L_Types(s, len, buckets, K, false, SA, t, cs);
            induce_S_Types(s, len, buckets, K, true, SA, t, cs);

            delete[] buckets;
            delete[] t;
            delete[] p_1;
        }
 
	    template<typename sa_index>
	    static void construct_sa(util::string_span text, size_t alphabet_size, util::span<sa_index> out_sa) {
            ssize* SA = new ssize[text.size()];	        
	        
            SAIS((unsigned char*)text.data(), SA, text.size(), alphabet_size, 1);
            for (size_t i = 0; i < text.size(); i++) { out_sa[i] = SA[i];} }   
	};
} 

