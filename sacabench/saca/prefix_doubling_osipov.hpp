#pragma once

#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/sort/stable_sort.hpp>
#include <util/sort/std_sort.hpp>
#include <util/sort/ips4o.hpp>
#include <algorithm>
#include <tuple>
#include <byteswap.h>
#include <tudocomp_stat/StatPhase.hpp>
#include <omp.h>

namespace sacabench::osipov {
    template<bool wordpacking_4_sort>
    class osipov_impl {
    public:
        template <typename sa_index>
        static void construct_sa(util::string_span text,
                                 util::span<sa_index> out_sa) {
            // Pretend we never even had the 8 extra bytes to begin with
            DCHECK_EQ(text.size(), out_sa.size());
            text = text.slice(0, text.size() - 8);
            out_sa = out_sa.slice(8, out_sa.size());

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
            inline bool operator()(const size_t& elem, const size_t& compare_to) const {
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
                const size_t& compare_to) const {

                if constexpr (wordpacking_4_sort) {
                    auto elem_wp = *((uint32_t const*) &text[elem]);
                    auto compare_to_wp = *((uint32_t const*) &text[compare_to]);
                    elem_wp = bswap_32(elem_wp);
                    compare_to_wp = bswap_32(compare_to_wp);

                    return elem_wp < compare_to_wp;
                } else {
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
                const std::tuple<sa_index, sa_index, sa_index>& compare_to) const {
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

        struct split_size_range {
            uint64_t start;
            uint64_t end;
        };

        static split_size_range split_size(size_t size, size_t thread_rank, size_t threads) {
            const uint64_t offset = ((size / threads) * thread_rank)
                + std::min<uint64_t>(thread_rank, size % threads);

            const uint64_t local_size = (size / threads)
                + ((thread_rank < size % threads) ? 1 : 0);

            return split_size_range { offset, offset + local_size };
        }

        template <typename sa_index, typename compare_func>
        static void initialize_isa(util::span<sa_index> sa,
            util::span<sa_index> isa, compare_func cmp) {
            // Sentinel has lowest rank
            isa[sa[0]] = static_cast<sa_index>(0);

            const uint64_t num_threads = omp_get_max_threads();

#pragma omp parallel
            {
                const uint64_t thread_id = omp_get_thread_num();
                //exclude first element (sentinel) \\v//
                auto range = split_size(sa.size() - 1, thread_id, num_threads);
                ++range.start;
                ++range.end;

                size_t i = range.start;

                // skip until start of first bucket
                for(; i < sa.size() && !(cmp(sa[i], sa[i-1]) || cmp(sa[i-1], sa[i])); ++i) {
                }

                // business as usual
                for(; i < range.end; ++i) {
                    if(!(cmp(sa[i], sa[i-1]) || cmp(sa[i-1], sa[i]))) {
                        isa[sa[i]] = isa[sa[i-1]];
                    } else {
                        isa[sa[i]] = i;
                    }
                }

#pragma omp barrier
                // continue until end of last bucket
                for(; i < sa.size() && isa[sa[i]] == static_cast<sa_index>(0); ++i) {
                    isa[sa[i]] = isa[sa[i-1]];
                }
            }
        }

        template <typename sa_index, typename compare_func>
        static void initialize_isa_aux(util::span<sa_index> sa,
            util::span<sa_index> isa, compare_func cmp) {

            util::container<sa_index> aux = util::make_container<sa_index>(sa.size());

            // Sentinel has lowest rank
            aux[0] = static_cast<sa_index>(0);
            isa[sa[0]] = static_cast<sa_index>(0);

            for (size_t i = 1; i < sa.size(); ++i) {
                if (!(cmp(sa[i-1], sa[i]) || cmp(sa[i-1], sa[i]))) {
                    aux[i] = static_cast<sa_index>(0);
                } else {
                    aux[i] = static_cast<sa_index>(i);
                }
            }

            for (size_t i = 1; i < sa.size(); ++i) {
                if (aux[i] > isa[sa[i-1]]) {
                    isa[sa[i]] = aux[i];
                } else {
                    isa[sa[i]] = isa[sa[i-1]];
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
            tdc::StatPhase phase("Initialization");

            // std::cout << "Starting Osipov sequential." << std::endl;
            // Check if enough bits free for negation.
            DCHECK(util::assert_text_length<sa_index>(text.size(), 1u));

            //std::cout << "Creating initial container." << std::endl;
            util::span<sa_index> sa = out_sa;
            auto isa_container = util::make_container<sa_index>(out_sa.size());
            util::span<sa_index> isa = util::span<sa_index>(isa_container);
            initialize_sa<sa_index>(text.size(), sa);

            sa_index h = 4;
            // Sort by h characters
            compare_first_four_chars cmp_init = compare_first_four_chars(text);
            phase.split("Initial 4-Sort");
            util::sort::ips4o_sort(sa, cmp_init);
            phase.split("Initialize ISA");
            initialize_isa<sa_index, compare_first_four_chars>(sa, isa, cmp_init);
            phase.split("Mark singletons");
            mark_singletons(sa, isa);
            phase.split("Loop Initialization");

            //std::cout << "isa: " << isa << std::endl;
            size_t size = sa.size();
            size_t s = 0;
            size_t index = 0;

            auto tuple_container = util::make_container<std::tuple<sa_index, sa_index, sa_index>>(size);
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
            compare_tuples<sa_index> cmp;
            while(size > 0) {
                phase.split("Iteration");

                //std::cout << "Elements left: " << size << std::endl;
                s=0;
                tuples = tuple_container.slice(0, size);
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
                    cmp = compare_tuples<sa_index>(tuples);
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
            phase.split("Write out SA");
            //std::cout << "Writing suffixes to out_sa. isa: " << isa << std::endl;
            for(size_t i=0; i < out_sa.size(); ++i) {
                out_sa[isa[i] ^ utils<sa_index>::NEGATIVE_MASK] = i;
            }
        }
    };
    struct osipov {
        static constexpr size_t EXTRA_SENTINELS = 1 + 8; // extra 8 to allow buffer overread during sorting
        static constexpr char const* NAME = "Osipov_sequential";
        static constexpr char const* DESCRIPTION =
            "Prefix Doubling approach for parallel gpu computation as sequential "
            "approach";


        template <typename sa_index>
        static void construct_sa(util::string_span text,
                                 util::alphabet const&,
                                 util::span<sa_index> out_sa) {
            osipov_impl<false>::construct_sa(text, out_sa);
        }
    };
    struct osipov_wp {
        static constexpr size_t EXTRA_SENTINELS = 1 + 8; // extra 8 to allow buffer overread during sorting
        static constexpr char const* NAME = "Osipov_sequential_wp";
        static constexpr char const* DESCRIPTION =
            "Prefix Doubling approach for parallel gpu computation as sequential "
            "approach";


        template <typename sa_index>
        static void construct_sa(util::string_span text,
                                 util::alphabet const&,
                                 util::span<sa_index> out_sa) {
            osipov_impl<true>::construct_sa(text, out_sa);
        }
    };
}
