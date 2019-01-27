#pragma once

#include <algorithm>
#include <tudocomp_stat/StatPhase.hpp>
#include <tuple>
#include <util/macros.hpp>
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

/*
    \brief Sequential implementation of the osipov algorithm used in class
    osipov (see osipov.hpp).
*/
template <bool wordpacking_4_sort, typename sa_index>
class osipov_seq{
private:
        osipov_spans<wordpacking_4_sort, sa_index> spans;


public:
    /*
        Constructor for class osipov_seq. Directly creates a osipov_spans
        instance within instantiation list (otherwise default constructor for
        osipov_spans is needed).

        @param out_sa Span for the final sa.
        @param isa Span for the isa.
        @param tuples Span for the tuples.
    */
    inline osipov_seq(util::span<sa_index> out_sa, util::span<sa_index> isa,
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples,
            util::string_span text) :
            spans(osipov_spans<wordpacking_4_sort, sa_index>(out_sa, isa, tuples, text)) {}

    // Returns the tuple span (needed for tuple comparison function)
    util::span<std::tuple<sa_index, sa_index, sa_index>> get_tuples() {
        return spans.tuples;
    }

    // Slices the tuples span to length end
    void slice_container(size_t end) {
        spans.slice_tuples(end);
    }

    /*
        \brief Updates the values for sa and isa given the values in tuples
        (called at end of iteration).

        @param s The new size for the next iteration
    */
    void update_container(size_t s) {
        // Update SA
        for (size_t i = 0; i < s; ++i) {
            spans.sa[i] = std::get<0>(spans.tuples[i]);
        }
        // Update ISA
        for (size_t i = 0; i < s; ++i) {
            spans.isa[std::get<0>(spans.tuples[i])] = std::get<1>(
                spans.tuples[i]);
        }

    }

    /*
        \brief Slices sa from 0 to end.
    */
    void slice_sa(size_t end) {spans.sa = spans.sa.slice(0, end);}

    /*
        \brief Writes final suffix array into out_sa.
    */
    void finalize(util::span<sa_index> out_sa) {spans.finalize(out_sa);}

    /*
        \brief Marks, i.e. inverts singleton h-groups in isa.
    */
    void mark_singletons() {
        auto sa = spans.sa;
        auto isa = spans.isa;
        // Only possible if there are still values in sa.
        if (sa.size() > 0) {
            // Creating container for flags
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

    /*
        \brief Sequentially initializes the isa.

        @param cmp The initial compare function (most probably
        compare_first_four_chars)
    */
    void initialize_isa() {
        auto sa = spans.sa;
        auto isa = spans.isa;
        auto cmp = spans.cmp_init;

        // Auxiliary array for marking new groups.
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

    /*
        \brief Uses ips4o as initial sorting function.

        @param cmp_init The compare function to sort by (most probably
        compare_first_four_chars)
    */
    void initial_sort() {
        util::sort::ips4o_sort(spans.sa, spans.cmp_init);
    }

    /*
        \brief Uses std::stable_sort as stable sorting method of tuples span in
        each iteration.

        @param cmp The compare function to sort the tuples.
    */
    void stable_sort() {
        util::sort::stable_sort(spans.tuples, spans.cmp_tuples);
    }

    /*
        \brief Generates tuples which are needed for current iteration.

        Generates tuples for suffixes still considered in this iteration as
        (suffix-index|h-rank|2h-rank) for suffixes which still need to be
        induced and as (suffix-index|h-rank|-h-rank) for suffixes which are
        already correctly sorted but are needed to induce other suffixes
        (because prefix of length h has already been considered for sorting).

        @param size The currently computed max size.
        @param h The depth of this iteration.
    */
    size_t create_tuples(size_t size, sa_index h) {
        auto sa = spans.sa;
        auto isa = spans.isa;
        auto tuples = spans.tuples;
        // s contains the amount of created tuples (initially 0)
        size_t s = 0;
        // Temporary index
        size_t index;
        for (size_t i = 0; i < size; ++i) {
            // Inducing suffix is still valid (i.e. suffix-index -
            // prefix-length >= 0)
            if (sa[i] >= h) {
                // Get index of considered suffix (getting induced by sa[i])
                index = sa[i] - h;
                // Considered suffix is not in singleton h-group -> create tuple
                if (((isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                     sa_index(0))) {
                    tuples[s++] =
                        std::make_tuple(index, isa[index], isa[sa[i]]);
                }
            }
            // Consider suffix sa[i] (may be needed in following iteration for
            // inducing)
            index = sa[i];
            /*
            sa[i] has singleton h-group AND induces valid index in next
            iteration AND this index has no singleton h-group (before this
            iteration) -> consider sa[i] in next iteration (as it will be able
            to induce another index).
            */
            if (((isa[index] & utils<sa_index>::NEGATIVE_MASK) > sa_index(0)) &&
                index >= 2 * h &&
                ((isa[index - 2 * h] & utils<sa_index>::NEGATIVE_MASK) ==
                 sa_index(0))) {
                tuples[s++] = std::make_tuple(
                    index, isa[index] ^ utils<sa_index>::NEGATIVE_MASK,
                    isa[index]);
            }
        }
        spans.set_cmp_tuples(tuples);
        // Return number of counted tuples.
        return s;
    }

    /*
        \brief Update rank of each tuple after tuples have been sorted.
    */
    void update_ranks(size_t) {
        auto tuples = spans.tuples;
        // Index for each reference point while updating index.
        sa_index head = 0;
        for (size_t i = 1; i < tuples.size(); ++i) {
            // Suffixes differ in h-rank -> different group, set new head.
            if (std::get<1>(tuples[i]) > std::get<1>(tuples[head])) {
                head = i;
            }
            // Same h-group, but differ in 2h-rank -> new h-rank for tuple i
            else if (std::get<2>(tuples[i]) !=
                       std::get<2>(tuples[head])) {
                tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                                            std::get<1>(tuples[head]) +
                                                sa_index(i) - head,
                                            std::get<2>(tuples[i]));
                head = i;
            }
            // Still same h-group -> copy h-rank of head in case of new rank for
            // head
            else {
                 tuples[i] = std::make_tuple(std::get<0>(tuples[i]),
                                            std::get<1>(tuples[head]),
                                            std::get<2>(tuples[i]));
            }
        }
    }
};

/*
    \brief Wrapper for calling sequential version of osipov.

    Wrapper class for calling the sequential version of the osipov algorithm
    which creates a valid osipov_seq instance and then calls the main osipov
    method. Template parameter wordpacking_4_sort specifies wether wordpacking
    is used while sa_index specifies the used type for suffix indices.
*/
template <bool wordpacking_4_sort, typename sa_index>
class osipov_impl_seq {
public:
    static void construct_sa(util::string_span text,
                             util::span<sa_index> out_sa) {
        // Pretend we never even had the 8 extra bytes to begin with
        DCHECK_EQ(text.size(), out_sa.size());
        text = text.slice(0, text.size() - 8);
        out_sa = out_sa.slice(8, out_sa.size());

        if (text.size() > 1) {
            // Create container for isa and tuples such that they remain valid
            // within whole computation of algorithm.
            auto isa_container = util::make_container<sa_index>(out_sa.size());
            auto tuple_container = util::make_container<std::tuple<sa_index,
                    sa_index, sa_index>>(out_sa.size());
            auto isa = util::span<sa_index>(isa_container);
            auto tuples = util::span<std::tuple<sa_index, sa_index, sa_index>>(
                    tuple_container);
            // create instance of sequential osipov implementation.
            auto impl = osipov_seq<wordpacking_4_sort, sa_index>(out_sa, isa, tuples, text);
            // Call main osipov function.
            osipov<sa_index>::prefix_doubling(text, out_sa, impl);
        } else {
            out_sa[0] = 0;
        }
    }
};

/*
    \brief Wrapper for calling sequential version of osipov (cpu) without
    wordpacking within framework.
*/
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

/*
    \brief Wrapper for calling sequential version of osipov (cpu) with
    wordpacking within framework.
*/
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
