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
    template <class osipov_impl>
    static void prefix_doubling(util::string_span text, util::span<sa_index> out_sa,
                osipov_impl& osipov) {
        tdc::StatPhase phase("Initialization");
        // std::cout << "Starting Osipov parallel." << std::endl;
        // Check if enough bits free for negation.
        DCHECK(util::assert_text_length<sa_index>(text.size(), 1u));

        sa_index h = 4;
        // Sort by h characters
        // TODO: Check if feasible for gpu-version
        auto cmp_init = compare_first_four_chars(text);
        phase.split("Initial 4-Sort");
        osipov.initial_sort(cmp_init);
        phase.split("Initialize ISA");
        osipov.initialize_isa(cmp_init);
        phase.split("Mark singletons");
        osipov.mark_singletons();
        phase.split("Loop Initialization");

        size_t size = osipov.get_size();
        size_t s = 0;

        // Could be needed for gpu-variant -> move to osipov_impl
        // TODO: Check if feasible for gpu-version
        compare_tuples cmp;
        while (size > 0) {
            phase.split("Iteration");
            // Using wrapper method
            osipov.slice_container(size);

            s = osipov.create_tuples(size, h);
            // Skip all operations till size gets its new size, if this
            // iteration contains no tuples
            if (s > 0) {
                // Wrapper for slicing tuple/aux
                osipov.slice_container(s);

                // Stably sort tuples
                auto tuples = osipov.get_tuples();
                cmp = compare_tuples(tuples);
                osipov.stable_sort(cmp);

                // Update ranks in tuples and adjust values to sa/isa
                osipov.slice_sa(s);
                osipov.update_ranks();
                osipov.update_container(s);
                osipov.mark_singletons();
            }

            size = s;
            h = 2 * h;
        }
        phase.split("Write out SA");
        osipov.finalize(out_sa);
    }
};
} // Namespace end
