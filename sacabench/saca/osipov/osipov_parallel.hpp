#pragma once

#include <algorithm>
#include <omp.h>
#include <tudocomp_stat/StatPhase.hpp>
#include <tuple>
#include <util/macros.hpp>
#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/bits.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/prefix_sum.hpp>
#include <util/sort/ips4o.hpp>
#include <util/sort/std_sort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <saca/osipov/osipov.hpp>
#include <util/uint_types.hpp>

namespace sacabench::osipov {


/*
    \brief Parallel implementation of the osipov algorithm used in class
    osipov (see osipov.hpp).
*/
template <bool wordpacking_4_sort, typename sa_index>
class osipov_par {
private:
    using aux_elem = util::next_primitive<sa_index>;

    osipov_spans<wordpacking_4_sort, sa_index> spans;
    util::span<aux_elem> aux;

public:
    /*
        Constructor for class osipov_par. Directly creates a osipov_spans
        instance within instantiation list (otherwise default constructor for
        osipov_spans is needed) and initializes aux.

        @param out_sa Span for the final sa.
        @param isa Span for the isa.
        @param tuples Span for the tuples.
        @param aux Span for auxiliary array.
    */
    inline osipov_par(util::span<sa_index> out_sa, util::span<sa_index> isa,
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples,
            util::span<aux_elem> aux, util::string_span text) :
            spans(osipov_spans<wordpacking_4_sort, sa_index>(out_sa, isa, tuples, text)), aux(aux){}

    // Returns the tuple span (needed for tuple comparison function)
    util::span<std::tuple<sa_index, sa_index, sa_index>> get_tuples() {
        return spans.tuples;
    }

    // Slices the tuples and aux span to length end
    void slice_container(size_t end) {
        spans.slice_tuples(end);
        aux = aux.slice(0, end);
    }

    /*
        \brief Updates the values for sa and isa given the values in tuples
        (called at end of iteration) in parallel.

        @param s The new size for the next iteration
    */
    void update_container(size_t s) {
        // Update SA
        #pragma omp parallel for if (s < 100)
        for (size_t i = 0; i < s; ++i) {
            spans.sa[i] = std::get<0>(spans.tuples[i]);
        }

        // Update ISA
        #pragma omp parallel for if (s < 100)
        for (size_t i = 0; i < s; ++i) {
            spans.isa[std::get<0>(spans.tuples[i])] =
                aux[i];
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
        \brief Parallel version for marking singleton h-groups, i.e. inverting
        their ranks using aux for flags to reduce memory usage.
        Parallelized via https://nvlabs.github.io/cub/classcub_1_1_block_discontinuity.html
    */
    void mark_singletons() {
        auto sa = spans.sa;
        auto isa = spans.isa;
        if (sa.size() > 0) {
            util::container<bool> flags = util::make_container<bool>(sa.size());
            flags[0] = true;

            // Set flags if predecessor has different rank.

            #pragma omp parallel for
            for (size_t i = 1; i < sa.size(); ++i) {
                flags[i] = (isa[sa[i - 1]] != isa[sa[i]]);
            }

            #pragma omp parallel for
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
                    isa[sa[sa.size() - 1]] | utils<sa_index>::NEGATIVE_MASK;
            }
        }
    }

    /*
        \brief Parallely initializes the isa.

        @param cmp The initial compare function (most probably
        compare_first_four_chars)
    */
    void initialize_isa() {
        auto sa = spans.sa;
        auto isa = spans.isa;
        auto cmp = spans.cmp_init;
        // Sentinel has lowest rank
        isa[sa[0]] = aux[0] = static_cast<sa_index>(0);

        #pragma omp parallel for
        for (size_t i = 1; i < sa.size(); ++i) {
            // no branching version of:
            // if in_same_bucket(sa[i-1], sa[i])
            //     aux[i] = 0;
            // else
            //     aux[i] = 1;
            aux[i] = static_cast<sa_index>(i) * (cmp(sa[i - 1], sa[i]) != 0);
        }

        for (size_t i = 1; i < sa.size(); ++i) {
            isa[sa[i]] = util::max<uint64_t>(aux[i], isa[sa[i - 1]]);
        }
    }

    template <typename elem_type, typename Op>
    void prefix_sum(util::span<elem_type> array, Op op) {
        // TODO replace with parallel
        for (size_t index = 1; index < array.size(); ++index) {
            array[index] = op(array[index], array[index - 1]);
        }
        /*
        //exclusive
        for(size_t index = array.size()-1;index>=1;--index)
        {
            array[index] = array[index-1];
        }
        array[0]=0;
        */
    }

    /*
        \brief Uses parallel ips4o as initial sorting function.

        @param cmp_init The compare function to sort by (most probably
        compare_first_four_chars)
    */
    void initial_sort() {
        util::sort::ips4o_sort_parallel(spans.sa, spans.cmp_init);
    }

    /*
        \brief Uses std::par_stable_sort as stable sorting method of tuples span
        in each iteration.

        @param cmp The compare function to sort the tuples.
    */
    void stable_sort() {
        util::sort::std_par_stable_sort(spans.tuples, spans.cmp_tuples);
    }

    /*
        \brief Generates tuples which are needed for current iteration.

        Generates tuples for suffixes still considered in this iteration as
        (suffix-index|h-rank|2h-rank) for suffixes which still need to be
        induced and as (suffix-index|h-rank|-h-rank) for suffixes which are
        already correctly sorted but are needed to induce other suffixes
        (because prefix of length h has already been considered for sorting).

        The parallel version first computes the positions parallely by setting
        values for the amount of created tuples (either 0, 1 or 2) in an
        auxiliary array aux and then computing the exclusive prefix sum over
        aux. After that the tuples can be created in parallel by using the
        computed values in aux (containing the position in the tuples array).

        @param size The currently computed max size.
        @param h The depth of this iteration.
    */
    size_t create_tuples(size_t size, sa_index h) {
        size_t s = 0;
        size_t index;
        #pragma omp parallel shared(aux) private(index)
        {
            // First pass: Set values in aux.
            #pragma omp for reduction(+ : s)
            for (size_t i = 0; i < size; ++i) {
                // Reset each value in aux to 0 (if tuples created: will be
                // increased)
                aux[i] = 0;
                if (spans.sa[i] >= h) {
                    index = spans.sa[i] - h;
                    if ((spans.isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                        sa_index(0)) {
                        // Critical environment because indexing doesn't
                        // work correctly otherwise (producing strange
                        // values in aux)

                        aux[i] = 1;
                    }
                }
                index = spans.sa[i];
                if (((spans.isa[index] & utils<sa_index>::NEGATIVE_MASK) >
                     sa_index(0)) &&
                    index >= 2 * h &&
                    ((spans.isa[index - 2 * h] & utils<sa_index>::NEGATIVE_MASK) ==
                     sa_index(0))) {
                    // Second condition met (i.e. another tuple) ->
                    // increase value of aux
                    // Critical environment: same as above

                    auto& data = aux[i];

                    #pragma omp atomic update
                    ++data;
                }
                s += aux[i];
            }

            // aux has been set, compute prefix sum
            #pragma omp single
            {
                util::seq_prefix_sum<aux_elem, util::sum_fct<aux_elem>>(
                    aux, aux, false, util::sum_fct<aux_elem>(), 0);
                // Contains position for first tuple created due to index at
                // position sa[aux.size()-1]
            }

            // Create tuples in parallel using positions computed in aux.
            #pragma omp for
            for (size_t i = 0; i < size; ++i) {
                // Create tuple using first condition and value in aux (increase
                // aux in case that second condition is met for i aswell)
                if (spans.sa[i] >= h) {
                    index = spans.sa[i] - h;
                    if ((spans.isa[index] & utils<sa_index>::NEGATIVE_MASK) ==
                        sa_index(0)) {
                        spans.tuples[aux[i]++] =
                            std::make_tuple(index, spans.isa[index], spans.isa[spans.sa[i]]);
                    }
                }
                index = spans.sa[i];
                // Create tuple using second condition and value in aux
                if (((spans.isa[index] & utils<sa_index>::NEGATIVE_MASK) >
                     sa_index(0)) &&
                    index >= 2 * h &&
                    ((spans.isa[index - 2 * h] & utils<sa_index>::NEGATIVE_MASK) ==
                     sa_index(0))) {
                    spans.tuples[aux[i]] = std::make_tuple(
                        index, spans.isa[index] ^ utils<sa_index>::NEGATIVE_MASK,
                        spans.isa[index]);
                }
            }
        }
        spans.set_cmp_tuples(spans.tuples);
        return s;
    }

    /*
        \brief Update rank of each tuple after tuples have been sorted.

        Parallel version of updating each rank using prefix sums.
    */
    void update_ranks(size_t) {
        auto sorted_tuples = spans.tuples;
        aux[0] = 0;

        #pragma omp parallel for
        for (size_t index = 1; index < sorted_tuples.size(); ++index) {
            bool different = (std::get<1>(sorted_tuples[index - 1]) !=
                              std::get<1>(sorted_tuples[index]));
            aux[index] = different * index;
        }

        // Maybe TODO: replace with parallel
        prefix_sum(aux, [](auto a, auto b) { return util::max(a, b); });

        aux[0] = std::get<1>(sorted_tuples[0]);

        #pragma omp parallel for
        for (sa_index index = 1; index < sorted_tuples.size(); ++index) {
            // New Group
            bool new_group = (std::get<1>(sorted_tuples[index - sa_index(1)]) !=
                                  std::get<1>(sorted_tuples[index]) ||
                              std::get<2>(sorted_tuples[index - sa_index(1)]) !=
                                  std::get<2>(sorted_tuples[index]));
            aux[index] = new_group * ((std::get<1>(sorted_tuples[index])) +
                                      index - sa_index(aux[index]));
        }
        // Maybe TODO: replace with parallel
        prefix_sum(aux, [](auto a, auto b) { return util::max(a, b); });
    }

};

/*
    \brief Wrapper for calling parallel version of osipov (cpu).

    Wrapper class for calling the parallel version of the osipov algorithm on
    the cpu which creates a valid osipov_par instance and then calls the main
    osipov method. Template parameter wordpacking_4_sort specifies wether
    wordpacking is used while sa_index specifies the used type for suffix
    indices.
*/
template <bool wordpacking_4_sort, typename sa_index>
class osipov_impl_par {
    using aux_elem = util::next_primitive<sa_index>;
public:
    static void construct_sa(util::string_span text,
                             util::span<sa_index> out_sa) {
        // Pretend we never even had the 8 extra bytes to begin with
        DCHECK_EQ(text.size(), out_sa.size());
        text = text.slice(0, text.size() - 8);
        out_sa = out_sa.slice(8, out_sa.size());

        if (text.size() > 1) {
            // Create spans needed for computation (i.e. isa, tuples, aux).
            auto isa_container = util::make_container<sa_index>(out_sa.size());
            auto tuple_container = util::make_container<std::tuple<sa_index,
                    sa_index, sa_index>>(out_sa.size());
            auto aux_container = util::make_container<aux_elem>(out_sa.size());

            auto isa = isa_container.slice();
            auto tuples = tuple_container.slice();
            auto aux = aux_container.slice();
            // Create instance of parallel osipov implementation.
            auto impl = osipov_par<wordpacking_4_sort, sa_index>(out_sa, isa,
                tuples, aux, text);
            // Call main osipov function.
            osipov<sa_index>::prefix_doubling(text, out_sa, impl);
        } else {
            out_sa[0] = 0;
        }
    }
};

/*
    \brief Wrapper for calling parallel version of osipov (cpu) without
    wordpacking within framework.
*/
struct osipov_parallel {
    static constexpr size_t EXTRA_SENTINELS =
        1 + 8; // extra 8 to allow buffer overread during sorting
    static constexpr char const* NAME = "Osipov_parallel";
    static constexpr char const* DESCRIPTION =
        "Prefix Doubling approach for parallel gpu computation as parallel "
        "approach";

    template <typename sa_index>
    static void construct_sa(util::string_span text, util::alphabet const&,
                             util::span<sa_index> out_sa) {
        osipov_impl_par<false, sa_index>::construct_sa(text, out_sa);
    }
};

/*
    \brief Wrapper for calling parallel version of osipov (cpu) with
    wordpacking within framework.
*/
struct osipov_parallel_wp {
    static constexpr size_t EXTRA_SENTINELS =
        1 + 8; // extra 8 to allow buffer overread during sorting
    static constexpr char const* NAME = "Osipov_parallel_wp";
    static constexpr char const* DESCRIPTION =
        "Prefix Doubling approach for parallel gpu computation as parallel "
        "approach on cpu";

    template <typename sa_index>
    static void construct_sa(util::string_span text, util::alphabet const&,
                             util::span<sa_index> out_sa) {
        osipov_impl_par<true, sa_index>::construct_sa(text, out_sa);
    }
};
} // namespace sacabench::osipov
