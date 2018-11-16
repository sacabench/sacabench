#pragma once

#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/sort/stable_sort.hpp>
#include <algorithm>
#include <tuple>


namespace sacabench::osipov {
    class osipov {
    public:
        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const* NAME = "Osipov_sequential";
        static constexpr char const* DESCRIPTION =
            "Prefix Doubling approach for parallel gpu computation as sequential "
            "approach";


        template <typename sa_index>
        static void construct_sa(util::string_span text,
                                 util::alphabet const& alphabet,
                                 util::span<sa_index> out_sa) {
            // TODO: Fill me with my algorithm
            if(text.size()>1) {
                prefix_doubling_sequential(text, out_sa);
            } else {
                out_sa[0]=0;
            }
        }

    private:

        template <typename sa_index>
        struct utils{
            static constexpr sa_index NEGATIVE_MASK = size_t(1)
            << (sizeof(sa_index) * 8 - 1);
        };

        struct compare_first_char{
        public:

            inline compare_first_char(const util::string_span text) : text(text) {}

            // elem and compare_to need to be smaller than input.size()
            inline bool operator()(const size_t& elem, const size_t& compare_to) {
                return text[elem] < text[compare_to];
            }

        private:
            const util::string_span text;
        };

        struct compare_first_four_chars{
        public:
            inline compare_first_four_chars(const util::string_span text)
            : text(text) {}
            
            inline bool operator()(const size_t& elem, 
                const size_t& compare_to) {
                // max_length computation to ensure fail-safety (although should
                // never be exceeded due to sentinel as last char)
                size_t max_elem_length = std::min(text.size() - elem, size_t(4));
                size_t max_compare_to_length = 
                    std::min(text.size() - compare_to, size_t(4));
                size_t max_length = std::min(max_elem_length, 
                    max_compare_to_length);
                for(size_t i=0; i<max_length; ++i) {
                    if(text[elem+i] != text[compare_to+i]) {
                        // Chars differ -> either elem is smaller or not
                        return (text[elem+i] < text[compare_to+i] ? true : false);
                    }
                }
                // suffixes didn't differ within their first 4 chars.
                return false;
            }
        
        private:
            const util::string_span text;
        };
        
        
        template <typename sa_index>
        struct compare_tuples{
        public:
            inline compare_tuples(util::span<std::tuple<sa_index, sa_index,
                sa_index>>& tuples) : tuples(tuples) {}

            // Empty constructor used for temporary creation of compare function
            inline compare_tuples() {}

            inline bool operator()(const std::tuple<sa_index, sa_index, sa_index>& elem,
                const std::tuple<sa_index, sa_index, sa_index>& compare_to) {
                return std::get<1>(elem) < std::get<1>(compare_to);
            }

        private:
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
        };

        template <typename sa_index>
        static void mark_singletons(util::span<sa_index> sa, util::span<sa_index> isa) {
            if(sa.size()>0) {
                util::container<bool> flags = util::make_container<bool>(sa.size());
                flags[0] = true;
                // Set flags if predecessor has different rank.
                for(size_t i=1; i < sa.size(); ++i) {
                    flags[i] = isa[sa[i-1]] != isa[sa[i]] ? true : false;
                }
                for(size_t i=0; i < sa.size()-1; ++i) {
                    // flags corresponding to predecessor having a different rank, i.e.
                    // suffix sa[i] has different rank than its predecessor and successor.
                    if(flags[i] && flags[i+1]) {
                        isa[sa[i]] = isa[sa[i]] | utils<sa_index>::NEGATIVE_MASK;
                    }
                }
                // Check for last position - is singleton, if it has a different rank
                // than its predecessor (because of missing successor).
                if(flags[sa.size()-1]) {
                    isa[sa[sa.size()-1]] = isa[sa[sa.size()-1]] ^ utils<sa_index>::NEGATIVE_MASK;
                }
            }
        }

        template <typename sa_index, typename compare_func>
        static void initialize_isa(util::span<sa_index> sa, 
            util::span<sa_index> isa, compare_func cmp) {
            // Sentinel has lowest rank
            isa[sa[0]] = 0;
            for(size_t i=1; i < sa.size(); ++i) {
                if(!(cmp(sa[i], sa[i-1]) || cmp(sa[i-1], sa[i]))) {
                    isa[sa[i]] = isa[sa[i-1]];
                } else {
                    isa[sa[i]] = i;
                }
            }
        }
        
        // Fill sa with initial indices
        template <typename sa_index>
        static void initialize_sa(size_t text_length, util::span<sa_index> sa) {
            for(size_t i=0; i < text_length; ++i) {
                sa[i] = i;
            }
        }



        template <typename sa_index>
        static void prefix_doubling_sequential(util::string_span text,
                                 util::span<sa_index> out_sa) {
            // std::cout << "Starting Osipov sequential." << std::endl;
            // Check if enough bits free for negation.
            DCHECK(util::assert_text_length<sa_index>(text.size(), 1u));

            //std::cout << "Creating initial container." << std::endl;
            auto sa_container = util::make_container<sa_index>(out_sa.size());
            util::span<sa_index> sa = util::span<sa_index>(sa_container);
            auto isa_container = util::make_container<sa_index>(out_sa.size());
            util::span<sa_index> isa = util::span<sa_index>(isa_container);
            initialize_sa<sa_index>(text.size(), sa);
            
            sa_index h = 4;
            // Sort by h characters
            compare_first_four_chars cmp_init = compare_first_four_chars(text);
            util::sort::stable_sort(sa, cmp_init);
            initialize_isa<sa_index, compare_first_four_chars>(sa, isa, cmp_init);
            mark_singletons(sa, isa);

            //std::cout << "isa: " << isa << std::endl;
            size_t size = sa.size(), s, index;
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
            compare_tuples<sa_index> cmp;
            while(size > 0) {
                //std::cout << "Elements left: " << size << std::endl;
                s=0;
                auto tuple_container = util::make_container<std::tuple<sa_index, sa_index, sa_index>>(size);
                tuples = util::span<std::tuple<sa_index, sa_index, sa_index>>(tuple_container);
                //std::cout << "Creating tuple." << std::endl;
                for(size_t i=0; i < size; ++i) {
                    // equals sa[i] - h >= 0
                    if(sa[i]>=h) {
                        index = sa[i] - h;
                        //std::cout << "sa["<<i<<"]-h=" << index << std::endl;
                        if(((isa[index] & utils<sa_index>::NEGATIVE_MASK) == sa_index(0))) {
                            //std::cout << "Adding " << index << " to tuples." << std::endl;
                            tuples[s++] = std::make_tuple(index, isa[index], isa[sa[i]]);
                        }
                    }
                    index = sa[i];
                    //std::cout << "sa["<<i<<"]:" << index << std::endl;
                    if(((isa[index] & utils<sa_index>::NEGATIVE_MASK) > sa_index(0)) &&
                        index >= 2*h &&
                        ((isa[index - 2*h] & utils<sa_index>::NEGATIVE_MASK) == sa_index(0))) {
                        //std::cout << "Second condition met. Adding " << index << std::endl;
                        tuples[s++] = std::make_tuple(index, isa[index] ^ utils<sa_index>::NEGATIVE_MASK,
                            isa[index]);
                    }
                }
                //std::cout << "Next size: " << s << std::endl; 
                // Skip all operations till size gets its new size, if this 
                // iteration contains no tuples
                if(s>0) {
                    tuples = tuples.slice(0, s);
                    //std::cout << "Sorting tuples." << std::endl;
                    cmp = compare_tuples(tuples);
                    util::sort::stable_sort(tuples, cmp);
                    sa = sa.slice(0, s);
                    //std::cout << "Writing new order to sa." << std::endl;
                    for(size_t i=0; i < s; ++i) {
                        sa[i] = std::get<0>(tuples[i]);
                    }
                    //std::cout << "Refreshing ranks for tuples" << std::endl;
                    sa_index head = 0;
                    for(size_t i=1; i < s; ++i) {
                        if(std::get<1>(tuples[i]) > std::get<1>(tuples[head])) {head=i;}
                        else if(std::get<2>(tuples[i])!=std::get<2>(tuples[head])) {
                            tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                            std::get<1>(tuples[head])+sa_index(i)-head, std::get<2>(tuples[i]));
                            head=i;
                        } else {
                            tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                            std::get<1>(tuples[head]), std::get<2>(tuples[i]));
                        }
                    }
                    //std::cout << "Setting new ranks in isa" << std::endl;
                    for(size_t i=0; i < s; ++i) {
                        //std::cout << "Assigning suffix " << std::get<0>(tuples[i]) 
                        //<< " rank " << std::get<1>(tuples[i]) << std::endl;
                        isa[std::get<0>(tuples[i])] = std::get<1>(tuples[i]);
                    }
                    //std::cout << "marking singleton groups." << std::endl;
                    mark_singletons(sa, isa);
                }
                size = s;
                h= 2*h;
            }
            //std::cout << "Writing suffixes to out_sa. isa: " << isa << std::endl;
            for(size_t i=0; i < out_sa.size(); ++i) {
                out_sa[isa[i] ^ utils<sa_index>::NEGATIVE_MASK] = i;
            }
        }
    };
}
