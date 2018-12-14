#pragma once

#include <tudocomp_stat/StatPhase.hpp>
#include <tuple>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>



namespace sacabench::osipov {

template <typename sa_index>
struct utils {
    static constexpr sa_index NEGATIVE_MASK = size_t(1)
                                              << (sizeof(sa_index) * 8 - 1);
};

template <typename sa_index>
struct osipov_spans {
    util::span<sa_index> sa;
    util::span<sa_index> isa;
    util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;

    inline osipov_spans(util::span<sa_index> out_sa, util::span<sa_index> isa,
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples) :
            sa(out_sa), isa(isa), tuples(tuples){
        initialize_sa(out_sa.size());
    }

    void initialize_sa(size_t text_length) {
        for (size_t i = 0; i < text_length; ++i) {
            sa[i] = i;
        }
    }


    // slicing tuples-span from 0 (begin) to end
    inline void slice_tuples(size_t end) {
        tuples = tuples.slice(0, end);
    }

    inline void finalize(util::span<sa_index> out_sa) {
        for (size_t i = 0; i < out_sa.size(); ++i) {
            out_sa[isa[i] ^ utils<sa_index>::NEGATIVE_MASK] = i;
        }
    }
};

template <bool wordpacking_4_sort, typename sa_index>
class osipov {

    struct compare_first_char {
    public:
        inline compare_first_char(const util::string_span text) : text(text) {}

        // elem and compare_to need to be smaller than input.size()
        inline bool operator()(const size_t& elem,
                               const size_t& compare_to) const {
            return text[elem] < text[compare_to];
        }

    private:
        const util::string_span text;
    };

    // template <bool wordpacking_4_sort>
    struct compare_first_four_chars {
    public:
        inline compare_first_four_chars(const util::string_span text)
            : text(text) {}

        inline bool operator()(const size_t& elem,
                               const size_t& compare_to) const {

            if constexpr (wordpacking_4_sort) {
                auto elem_wp = *((uint32_t const*)&text[elem]);
                auto compare_to_wp = *((uint32_t const*)&text[compare_to]);
                elem_wp = bswap_32(elem_wp);
                compare_to_wp = bswap_32(compare_to_wp);

                return elem_wp < compare_to_wp;
            } else {
                // max_length computation to ensure fail-safety (although should
                // never be exceeded due to sentinel as last char)
                size_t max_elem_length =
                    std::min(text.size() - elem, size_t(4));
                size_t max_compare_to_length =
                    std::min(text.size() - compare_to, size_t(4));
                size_t max_length =
                    std::min(max_elem_length, max_compare_to_length);
                for (size_t i = 0; i < max_length; ++i) {
                    if (text[elem + i] != text[compare_to + i]) {
                        // Chars differ -> either elem is smaller or not
                        return (text[elem + i] < text[compare_to + i]);
                    }
                }

                // suffixes didn't differ within their first 4 chars.
                return false;
            }
        }

    private:
        const util::string_span text;
    };

    // template <typename sa_index>
    struct compare_tuples {
    public:
        inline compare_tuples(
            util::span<std::tuple<sa_index, sa_index, sa_index>>& tuples)
            : tuples(tuples) {}

        // Empty constructor used for temporary creation of compare function
        inline compare_tuples() {}

        inline bool operator()(
            const std::tuple<sa_index, sa_index, sa_index>& elem,
            const std::tuple<sa_index, sa_index, sa_index>& compare_to) const {
            return std::get<1>(elem) < std::get<1>(compare_to);
        }

    private:
        util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
    };

public:
    /*
        template <typename sa_index, typename update_ranks_func,
                typename create_tuples_func, typename init_isa_func,
                typename init_sort_func, typename stable_sort_func>
        static void prefix_doubling(util::string_span text,
                util::span<sa_index> out_sa, update_ranks_func update_ranks,
                create_tuples_func create_tuples, init_isa_func initialize_isa,
                init_sort_func initial_sort, stable_sort_func stable_sort) {
                */
    // template <typename sa_index, typename osipov_impl>
    template <class osipov_impl>
    static void prefix_doubling(util::string_span text, util::span<sa_index> out_sa,
                osipov_impl& osipov) {
        tdc::StatPhase phase("Initialization");
        // std::cout << "Starting Osipov parallel." << std::endl;
        // Check if enough bits free for negation.
        DCHECK(util::assert_text_length<sa_index>(text.size(), 1u));

        // std::cout << "Creating initial container." << std::endl;

        // Export container to structs; initializations methods of struct
        /*
        util::span<sa_index> sa = out_sa;
        auto isa_container = util::make_container<sa_index>(out_sa.size());
        auto aux_container = util::make_container<sa_index>(out_sa.size());

        util::span<sa_index> isa = util::span<sa_index>(isa_container);
        util::span<sa_index> aux = util::span<sa_index>(aux_container);
        */
        // Init with out_sa (mostly needed for its size)

        // initialize_sa<sa_index>(text.size(), sa);

        sa_index h = 4;
        // Sort by h characters
        // TODO: Check if feasible for gpu-version
        auto cmp_init = compare_first_four_chars(text);
        phase.split("Initial 4-Sort");
        osipov.initial_sort(cmp_init);
        // util::sort::ips4o_sort_parallel(sa, cmp_init);
        phase.split("Initialize ISA");
        osipov.initialize_isa(cmp_init);
        phase.split("Mark singletons");
        osipov.mark_singletons();
        phase.split("Loop Initialization");

        // std::cout << "isa: " << isa << std::endl;
        size_t size = osipov.get_size();
        size_t s = 0;

        // Could be needed for gpu-variant -> move to osipov_impl
        // (init with first init)
        /*auto tuple_container =
            util::make_container<std::tuple<sa_index, sa_index, sa_index>>(
                size);
        util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;*/
        // TODO: Check if feasible for gpu-version
        compare_tuples cmp;
        while (size > 0) {
            phase.split("Iteration");
            // Using wrapper method
            /*aux = util::span<sa_index>(aux_container).slice(0, size);
            tuples = tuple_container.slice(0, size);
            */
            osipov.slice_container(size);

            // s = create_tuples<sa_index>(tuples.slice(0, size), size, h, sa,
            // isa);
            s = osipov.create_tuples(size, h);
            // std::cout << "Elements left: " << size << std::endl;

            // std::cout << "Next size: " << s << std::endl;
            // Skip all operations till size gets its new size, if this
            // iteration contains no tuples
            if (s > 0) {
                // Wrapper for slicing tuple/aux
                osipov.slice_container(s);
                /*
                tuples = tuples.slice(0, s);
                aux = util::span<sa_index>(aux).slice(0, s); */
                auto tuples = osipov.get_tuples();
                cmp = compare_tuples(tuples);
                osipov.stable_sort(cmp);
                // util::sort::std_par_stable_sort(tuples, cmp);

                // osipov.sa = osipov.sa.slice(0, s);
                osipov.slice_sa(s);
                osipov.update_ranks();
                // std::cout << "Writing new order to sa." << std::endl;

                // Update values in sa and isa after correct rank computation
                // in 'update_ranks' (maybe include it there?)
                //osipov.update_container(s);
                osipov.update_sa(s);
                osipov.update_isa(s);
                /*
                #pragma omp parallel for if (s < 100)
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

                #pragma omp parallel for if (s < 100)
                for (size_t i = 0; i < s; ++i) {
                    // std::cout << "Assigning suffix " <<
                    // std::get<0>(tuples[i])
                    //<< " rank " << std::get<1>(tuples[i]) << std::endl;
                    isa[std::get<0>(tuples[i])] =
                        aux[i]; // std::get<1>(tuples[i]);
                }*/
                osipov.mark_singletons();
            }

            size = s;
            h = 2 * h;
        }
        phase.split("Write out SA");
        // std::cout << "Writing suffixes to out_sa. isa: " << isa << std::endl;
        osipov.finalize(out_sa);
    }
};
} // Namespace end
