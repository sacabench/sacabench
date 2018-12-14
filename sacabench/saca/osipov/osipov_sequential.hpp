#pragma once

#include <algorithm>
#include <byteswap.h>
#include <tudocomp_stat/StatPhase.hpp>
#include <tuple>
#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/sort/ips4o.hpp>
#include <util/sort/stable_sort.hpp>
#include <util/sort/std_sort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <saca/osipov/osipov.hpp>

namespace sacabench::osipov {


template <typename sa_index>
class osipov_seq{
private:
        osipov_spans<sa_index> spans;


public:
    inline osipov_seq(util::span<sa_index> out_sa, util::span<sa_index> isa,
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples) :
            spans(osipov_spans<sa_index>(out_sa, isa, tuples)) {
        //spans = osipov_spans<sa_index>(out_sa);
    }

    size_t get_size(){return spans.sa.size();}

    util::span<std::tuple<sa_index, sa_index, sa_index>> get_tuples() {
        return spans.tuples;
    }

    void slice_container(size_t end) {
        spans.slice_tuples(end);
    }

    void update_container(size_t s) {
        // Update SA
        for (size_t i = 0; i < s; ++i) {
            spans.sa[i] = std::get<0>(spans.tuples[i]);
        }

        // Update ISA
        for (size_t i = 0; i < s; ++i) {
            // std::cout << "Assigning suffix " <<
            // std::get<0>(tuples[i])
            //<< " rank " << std::get<1>(tuples[i]) << std::endl;
            spans.isa[std::get<0>(spans.tuples[i])] = std::get<1>(
                spans.tuples[i]);
        }
    }

    void slice_sa(size_t end) {spans.sa.slice(0, end);}

    void finalize(util::span<sa_index> out_sa) {spans.finalize(out_sa);}

    // template <typename sa_index>
    void mark_singletons() {
        auto sa = spans.sa;
        auto isa = spans.isa;
        if (sa.size() > 0) {
            util::container<bool> flags = util::make_container<bool>(sa.size());
            flags[0] = true;
            // Set flags if predecessor has different rank.
            for (size_t i = 1; i < sa.size(); ++i) {
                flags[i] = isa[sa[i - 1]] != isa[sa[i]] ? true : false;
            }
            for (size_t i = 0; i < sa.size() - 1; ++i) {
                // flags corresponding to predecessor having a different rank,
                // i.e. suffix sa[i] has different rank than its predecessor and
                // successor.
                if (flags[i] && flags[i + 1]) {
                    isa[sa[i]] = isa[sa[i]] | utils<sa_index>::NEGATIVE_MASK;
                }
            }
            // Check for last position - is singleton, if it has a different
            // rank than its predecessor (because of missing successor).
            if (flags[sa.size() - 1]) {
                isa[sa[sa.size() - 1]] =
                    isa[sa[sa.size() - 1]] ^ utils<sa_index>::NEGATIVE_MASK;
            }
        }
    }

    // Sequential variant of initializing the isa
    // template <typename sa_index, typename compare_func>
    template <typename compare_func>
    void initialize_isa(compare_func cmp) {
        auto sa = spans.sa;
        auto isa = spans.isa;

        util::container<sa_index> aux =
            util::make_container<sa_index>(sa.size());


        // Sentinel has lowest rank
        isa[sa[0]] = aux[0] = static_cast<sa_index>(0);

        for (size_t i = 1; i < sa.size(); ++i) {
            // no branching version of:
            // if in_same_bucket(sa[i-1], sa[i])
            //     aux[i] = 0;
            // else
            //     aux[i] = 1;
            aux[i] = static_cast<sa_index>(i) *
                     ((cmp(sa[i - 1], sa[i]) | cmp(sa[i - 1], sa[i])) != 0);
        }

        for (size_t i = 1; i < sa.size(); ++i) {
            isa[sa[i]] = aux[i] > isa[sa[i - 1]] ? aux[i] : isa[sa[i - 1]];
        }
    }

    // Fill sa with initial indices
    // template <typename sa_index>
    void initialize_sa(size_t text_length) {
        for (size_t i = 0; i < text_length; ++i) {
            spans.sa[i] = i;
        }
    }

    template <typename compare_func>
    void initial_sort(compare_func cmp_init) {
        util::sort::ips4o_sort(spans.sa, cmp_init);
    }

    template <typename compare_func>
    void stable_sort(compare_func cmp) {
        util::sort::stable_sort(spans.tuples, cmp);
    }


    // template <typename sa_index>
    size_t create_tuples(size_t size, sa_index h) {
        auto sa = spans.sa;
        auto isa = spans.isa;
        auto tuples = spans.tuples;
        size_t s = 0;
        size_t index;
        // std::cout << "Creating tuple." << std::endl;
        for (size_t i = 0; i < size; ++i) {
            // equals sa[i] - h >= 0
            if (sa[i] >= h) {
                index = sa[i] - h;
                // std::cout << "sa["<<i<<"]-h=" << index << std::endl;
                if (((isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                     sa_index(0))) {
                    // std::cout << "Adding " << index << " to tuples." <<
                    // std::endl;
                    tuples[s++] =
                        std::make_tuple(index, isa[index], isa[sa[i]]);
                    /*std::cout << "(" << index << "|" << isa[index] << "|"
                    << isa[sa[i]] << ")" << std::endl;*/
                }
            }
            index = sa[i];
            // std::cout << "sa["<<i<<"]:" << index << std::endl;
            if (((isa[index] & utils<sa_index>::NEGATIVE_MASK) > sa_index(0)) &&
                index >= 2 * h &&
                ((isa[index - 2 * h] & utils<sa_index>::NEGATIVE_MASK) ==
                 sa_index(0))) {
                // std::cout << "Second condition met. Adding " << index <<
                // std::endl;
                tuples[s++] = std::make_tuple(
                    index, isa[index] ^ utils<sa_index>::NEGATIVE_MASK,
                    isa[index]);
                /*std::cout << "(" << index << "|"
                << (isa[index] ^ utils<sa_index>::NEGATIVE_MASK) << "|"
                << isa[index] << ")" << std::endl;*/
            }
        }
        return s;
    }

    void update_ranks() {
        auto tuples = spans.tuples;
        sa_index head = 0;
        for (size_t i = 1; i < tuples.size(); ++i) {
            if (std::get<1>(tuples[i]) > std::get<1>(tuples[head])) {
                head = i;
            } else if (std::get<2>(tuples[i]) !=
                       std::get<2>(tuples[head])) {
                tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                                            std::get<1>(tuples[head]) +
                                                sa_index(i) - head,
                                            std::get<2>(tuples[i]));
                head = i;
            } else {
                tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                                            std::get<1>(tuples[head]),
                                            std::get<2>(tuples[i]));
            }
        }
    }


/*
    //template <typename sa_index>
    static void prefix_doubling_sequential(util::string_span text,
                                           util::span<sa_index> out_sa) {
        tdc::StatPhase phase("Initialization");

        // std::cout << "Starting Osipov sequential." << std::endl;
        // Check if enough bits free for negation.
        DCHECK(util::assert_text_length<sa_index>(text.size(), 1u));

        // std::cout << "Creating initial container." << std::endl;
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

        // std::cout << "isa: " << isa << std::endl;
        size_t size = sa.size();
        size_t s = 0;
        size_t index = 0;

        auto tuple_container =
            util::make_container<std::tuple<sa_index, sa_index, sa_index>>(
                size);
        util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
        osipov<>compare_tuples<sa_index> cmp;
        while (size > 0) {
            phase.split("Iteration");

            // std::cout << "Elements left: " << size << std::endl;
            s = 0;
            tuples = tuple_container.slice(0, size);
            // std::cout << "Creating tuple." << std::endl;
            for (size_t i = 0; i < size; ++i) {
                // equals sa[i] - h >= 0
                if (sa[i] >= h) {
                    index = sa[i] - h;
                    // std::cout << "sa["<<i<<"]-h=" << index << std::endl;
                    if (((isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                         sa_index(0))) {
                        // std::cout << "Adding " << index << " to tuples." <<
                        // std::endl;
                        tuples[s++] =
                            std::make_tuple(index, isa[index], isa[sa[i]]);
                    }
                }
                index = sa[i];
                // std::cout << "sa["<<i<<"]:" << index << std::endl;
                if (((isa[index] & utils<sa_index>::NEGATIVE_MASK) >
                     sa_index(0)) &&
                    index >= 2 * h &&
                    ((isa[index - 2 * h] & utils<sa_index>::NEGATIVE_MASK) ==
                     sa_index(0))) {
                    // std::cout << "Second condition met. Adding " << index <<
                    // std::endl;
                    tuples[s++] = std::make_tuple(
                        index, isa[index] ^ utils<sa_index>::NEGATIVE_MASK,
                        isa[index]);
                }
            }
            // std::cout << "Next size: " << s << std::endl;
            // Skip all operations till size gets its new size, if this
            // iteration contains no tuples
            if (s > 0) {
                tuples = tuples.slice(0, s);
                // std::cout << "Sorting tuples." << std::endl;
                cmp = osipov::compare_tuples<sa_index>(tuples);
                util::sort::stable_sort(tuples, cmp);
                sa = sa.slice(0, s);
                // std::cout << "Writing new order to sa." << std::endl;
                for (size_t i = 0; i < s; ++i) {
                    sa[i] = std::get<0>(tuples[i]);
                }
                // std::cout << "Refreshing ranks for tuples" << std::endl;
                sa_index head = 0;
                for (size_t i = 1; i < s; ++i) {
                    if (std::get<1>(tuples[i]) > std::get<1>(tuples[head])) {
                        head = i;
                    } else if (std::get<2>(tuples[i]) !=
                               std::get<2>(tuples[head])) {
                        tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                                                    std::get<1>(tuples[head]) +
                                                        sa_index(i) - head,
                                                    std::get<2>(tuples[i]));
                        head = i;
                    } else {
                        tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                                                    std::get<1>(tuples[head]),
                                                    std::get<2>(tuples[i]));
                    }
                }
                // std::cout << "Setting new ranks in isa" << std::endl;
                for (size_t i = 0; i < s; ++i) {
                    // std::cout << "Assigning suffix " <<
                    // std::get<0>(tuples[i])
                    //<< " rank " << std::get<1>(tuples[i]) << std::endl;
                    isa[std::get<0>(tuples[i])] = std::get<1>(tuples[i]);
                }
                // std::cout << "marking singleton groups." << std::endl;
                mark_singletons(sa, isa);
            }
            size = s;
            h = 2 * h;
        }
        phase.split("Write out SA");
        // std::cout << "Writing suffixes to out_sa. isa: " << isa << std::endl;
        for (size_t i = 0; i < out_sa.size(); ++i) {
            out_sa[isa[i] ^ utils<sa_index>::NEGATIVE_MASK] = i;
        }
    }*/
};

template <bool wordpacking_4_sort, typename sa_index>
class osipov_impl_seq {
public:
    // template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::span<sa_index> out_sa) {
        // Pretend we never even had the 8 extra bytes to begin with
        DCHECK_EQ(text.size(), out_sa.size());
        text = text.slice(0, text.size() - 8);
        out_sa = out_sa.slice(8, out_sa.size());

        if (text.size() > 1) {
            auto isa_container = util::make_container<sa_index>(out_sa.size());
            auto tuple_container = util::make_container<std::tuple<sa_index,
                    sa_index, sa_index>>(out_sa.size());
            auto isa = util::span<sa_index>(isa_container);
            auto tuples = util::span<std::tuple<sa_index, sa_index, sa_index>>(
                    tuple_container);
            auto impl = osipov_seq<sa_index>(out_sa, isa, tuples);
            osipov<wordpacking_4_sort, sa_index>::prefix_doubling(text, out_sa,
                impl);
        } else {
            out_sa[0] = 0;
        }
    }
};

struct osipov_sequential {
    static constexpr size_t EXTRA_SENTINELS =
        1 + 8; // extra 8 to allow buffer overread during sorting
    static constexpr char const* NAME = "Osipov_sequential";
    static constexpr char const* DESCRIPTION =
        "Prefix Doubling approach for parallel gpu computation as sequential "
        "approach";

    template <typename sa_index>
    static void construct_sa(util::string_span text, util::alphabet const&,
                             util::span<sa_index> out_sa) {
        osipov_impl_seq<false, sa_index>::construct_sa(text, out_sa);
    }
};
struct osipov_sequential_wp {
    static constexpr size_t EXTRA_SENTINELS =
        1 + 8; // extra 8 to allow buffer overread during sorting
    static constexpr char const* NAME = "Osipov_sequential_wp";
    static constexpr char const* DESCRIPTION =
        "Prefix Doubling approach for parallel gpu computation as sequential "
        "approach";

    template <typename sa_index>
    static void construct_sa(util::string_span text, util::alphabet const&,
                             util::span<sa_index> out_sa) {
        osipov_impl_seq<true, sa_index>::construct_sa(text, out_sa);
    }
};
} // namespace sacabench::osipov
