#pragma once

#include <algorithm>
#include <byteswap.h>
#include <omp.h>
#include <tudocomp_stat/StatPhase.hpp>
#include <tuple>
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

namespace sacabench::osipov {

template <typename sa_index>
class osipov_par {
private:
    osipov_spans<sa_index> spans;
    util::span<sa_index> aux;

public:
    // Constructor for osipov_par creating a osipov_spans instance and an
    // aux-container
    inline osipov_par(util::span<sa_index> out_sa, util::span<sa_index> isa,
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples,
            util::span<sa_index> aux) :
            spans(osipov_spans<sa_index>(out_sa, isa, tuples)), aux(aux){}



    size_t get_size() {return spans.sa.size();}

    util::span<std::tuple<sa_index, sa_index, sa_index>> get_tuples() {
        return spans.tuples;
    }

    void slice_container(size_t end) {
        spans.slice_tuples(end);
        aux = aux.slice(0, end);
    }

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

    void slice_sa(size_t end) {spans.sa = spans.sa.slice(0, end);}

    void finalize(util::span<sa_index> out_sa) {spans.finalize(out_sa);}

    /*
      Parallel version using aux for flags to reduce memory usage
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

    template <typename compare_func>
    void initialize_isa(compare_func cmp) {
        auto sa = spans.sa;
        auto isa = spans.isa;
        // Sentinel has lowest rank
        isa[sa[0]] = aux[0] = static_cast<sa_index>(0);

        #pragma omp parallel for
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
            isa[sa[i]] = util::max(aux[i], isa[sa[i - 1]]);
        }
    }

    // Fill sa with initial indices
    void initialize_sa(size_t text_length) {
        for (size_t i = 0; i < text_length; ++i) {
            spans.sa[i] = i;
        }
    }

    template <typename Op>
    void prefix_sum(util::span<sa_index> array, Op op) {
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

    template <typename compare_func>
    void initial_sort(compare_func cmp_init) {
        util::sort::ips4o_sort_parallel(spans.sa, cmp_init);
    }

    template <typename compare_func>
    void stable_sort(compare_func cmp) {
        util::sort::std_par_stable_sort(spans.tuples, cmp);
    }

    size_t create_tuples(size_t size, sa_index h) {
        size_t s = 0;
        size_t index;
        #pragma omp parallel shared(aux) private(index)
        {

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

                        #pragma omp critical(aux)
                        {
                            ++aux[i];
                        }
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

                    #pragma omp critical(aux)
                    {
                        ++aux[i];
                    }
                }
                s += aux[i];
            }
            // aux has been set, compute prefix left_sum

            #pragma omp single
            {
                // Last value gets overwritten (exclusive prefix sum) ->
                // save value in s.

                util::seq_prefix_sum<sa_index, util::sum_fct<sa_index>>(
                    aux, aux, false, util::sum_fct<sa_index>(), 0);
                // Contains position for first tuple created due to index at
                // position sa[aux.size()-1]
            }

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
        return s;
    }

    void update_ranks() {
        auto sorted_tuples = spans.tuples;
        aux[0] = 0;

        #pragma omp parallel for
        for (size_t index = 1; index < sorted_tuples.size(); ++index) {
            bool different = (std::get<1>(sorted_tuples[index - 1]) !=
                              std::get<1>(sorted_tuples[index]));
            aux[index] = different * index;
        }

        // Maybe TODO: replace with parallel
        prefix_sum(aux, [](sa_index a, sa_index b) { return util::max(a, b); });

        aux[0] = std::get<1>(sorted_tuples[0]);

        #pragma omp parallel for
        for (sa_index index = 1; index < sorted_tuples.size(); ++index) {
            // New Group
            bool new_group = (std::get<1>(sorted_tuples[index - sa_index(1)]) !=
                                  std::get<1>(sorted_tuples[index]) ||
                              std::get<2>(sorted_tuples[index - sa_index(1)]) !=
                                  std::get<2>(sorted_tuples[index]));
            aux[index] = new_group * ((std::get<1>(sorted_tuples[index])) +
                                      index - aux[index]);
        }
        // Maybe TODO: replace with parallel
        prefix_sum(aux, [](sa_index a, sa_index b) { return util::max(a, b); });
    }

};
template <bool wordpacking_4_sort, typename sa_index>
class osipov_impl_par {
public:
    static void construct_sa(util::string_span text,
                             util::span<sa_index> out_sa) {
        // Pretend we never even had the 8 extra bytes to begin with
        DCHECK_EQ(text.size(), out_sa.size());
        text = text.slice(0, text.size() - 8);
        out_sa = out_sa.slice(8, out_sa.size());

        if (text.size() > 1) {
            // Create spans needed for computation
            auto isa_container = util::make_container<sa_index>(out_sa.size());
            auto tuple_container = util::make_container<std::tuple<sa_index,
                    sa_index, sa_index>>(out_sa.size());
            auto aux_container = util::make_container<sa_index>(out_sa.size());
            auto isa = util::span<sa_index>(isa_container);
            auto tuples = util::span<std::tuple<sa_index, sa_index, sa_index>>(
                    tuple_container);
            auto aux = util::span<sa_index>(aux_container);
            auto impl = osipov_par<sa_index>(out_sa, isa, tuples, aux);
            osipov<wordpacking_4_sort,sa_index>::prefix_doubling(text, out_sa,
                impl);
        } else {
            out_sa[0] = 0;
        }
    }
};

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
struct osipov_parallel_wp {
    static constexpr size_t EXTRA_SENTINELS =
        1 + 8; // extra 8 to allow buffer overread during sorting
    static constexpr char const* NAME = "Osipov_parallel_wp";
    static constexpr char const* DESCRIPTION =
        "Prefix Doubling approach for parallel gpu computation as parallel "
        "approach";

    template <typename sa_index>
    static void construct_sa(util::string_span text, util::alphabet const&,
                             util::span<sa_index> out_sa) {
        osipov_impl_par<true, sa_index>::construct_sa(text, out_sa);
    }
};
} // namespace sacabench::osipov
