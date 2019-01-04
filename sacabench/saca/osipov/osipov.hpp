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

/*
    \brief Comparison function comparing only the first symbol of two
    suffixes. Used for initial sort of sa.

    @param text Text the suffix array is being constructed for.
*/
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

/*
    \brief Comparison function comparing the first four symbols of two
    suffixes. Depentant on the generic bool parameter (wordpacking_4_sort)
    wordpacking is either used or not. Used for initial sort of sa.

    @param text Text the suffix array is being constructed for.
*/
template <bool wordpacking_4_sort>
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

/*
    \brief Comparison function for comparing two tuples used in the osipov
    algorithm.

    This comparison function takes the indices of two tuples within a span
    of all tuples and checks wether the second component is smaller for the
    first tuple (second component of a tuple is sort key for stable sort
    within each iteration).
*/
template <typename sa_index>
struct compare_tuples {
public:
    inline compare_tuples(
        util::span<std::tuple<sa_index, sa_index, sa_index>> tuples)
        : tuples(tuples) {}

    // Empty constructor used for temporary creation of compare function
    inline compare_tuples() {}

    inline bool operator()(
        const std::tuple<sa_index, sa_index, sa_index>& elem,
        const std::tuple<sa_index, sa_index, sa_index>& compare_to) const {
        return std::get<1>(elem) < std::get<1>(compare_to);
    }

    inline void set_tuples(util::span<std::tuple<sa_index, sa_index, sa_index>>
            new_tuples) {
        tuples = new_tuples;
    }

private:
    util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
};

/*
    Wrapper struct which contains spans for sa, isa and tuples.
*/
template <bool wordpacking_4_sort, typename sa_index>
struct osipov_spans {
    util::span<sa_index> sa;
    util::span<sa_index> isa;
    util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
    compare_first_four_chars<wordpacking_4_sort> cmp_init;
    compare_tuples<sa_index> cmp_tuples;

    /*
        \brief Constructor given spans for sa, isa and tuples. Initializes the
        sa with default values (0 to n-1).

        Creates an instance of osipov_spans given a span for the sa, isa and
        tuples each. After setting each variable, the sa gets initialized with
        default values.

        @param out_sa Span for sa indices
        @param isa Span for isa indices (i.e. ranks)
        @param tuples Span containing all tuples created within a loop iteration
        in the osipov algorithm.
    */
    inline osipov_spans(util::span<sa_index> out_sa, util::span<sa_index> isa,
            util::span<std::tuple<sa_index, sa_index, sa_index>> tuples,
            util::string_span text) : sa(out_sa), isa(isa), tuples(tuples),
            cmp_init(compare_first_four_chars<wordpacking_4_sort>(text)) {
        initialize_sa(out_sa.size());
    }

    // Initial values for sa (unsorted)
    void initialize_sa(size_t text_length) {
        for (size_t i = 0; i < text_length; ++i) {
            sa[i] = i;
        }
    }

    inline void set_cmp_tuples(util::span<std::tuple<sa_index, sa_index,
            sa_index>> tuples) {
        cmp_tuples.set_tuples(tuples);
    }


    // slicing tuples-span from 0 (begin) to end
    inline void slice_tuples(size_t end) {
        tuples = tuples.slice(0, end);
    }

    /*
        \brief Writes the final values for the sa given a final isa (within this
        instance)

        @param out_sa The span for the final sa with size n.
    */
    inline void finalize(util::span<sa_index> out_sa) {
        for (size_t i = 0; i < out_sa.size(); ++i) {
            /*
                Each singleton h-group's rank has been inverted. After final iteration,
                each suffix is in a singleton group, i.e. all ranks have been
                inverted.
            */
            out_sa[isa[i] ^ utils<sa_index>::NEGATIVE_MASK] = i;
        }
    }
};

/*
    \brief Class containing the comparison functions for the cpu-versions and
    the main method frame for the osipov algorithm, given a specific
    implementation of the algorithm (containing all methods being called here).
*/
template <typename sa_index>
class osipov {



/*
    \brief Main function of osipov algorithm. Takes an osipov implementation
    which supplies all needed methods within this algorithm.

    @param text Text the suffix array is being computed for.
    @param out_sa Span for the output sa.
    @param osipov The implementation used within this osipov algorithm.
*/
public:
    template <class osipov_impl>
    static void prefix_doubling(util::string_span text, util::span<sa_index> out_sa,
                osipov_impl& osipov) {
        tdc::StatPhase phase("Initialization");
        // Check if enough bits free for negation.
        DCHECK(util::assert_text_length<sa_index>(text.size(), 1u));

        sa_index h = 4;
        // Sort by h characters
        // TODO: Check if feasible for gpu-version
        phase.split("Initial 4-Sort");
        osipov.initial_sort();
        phase.split("Initialize ISA");
        osipov.initialize_isa();
        phase.split("Mark singletons");
        osipov.mark_singletons();
        phase.split("Loop Initialization");

        // Initial size containing all suffixes
        size_t size = out_sa.size();
        size_t s = 0;

        // Could be needed for gpu-variant -> move to osipov_impl
        // TODO: Check if feasible for gpu-version
        while (size > 0) {
            phase.split("Iteration");
            // Using wrapper method for slicing all corresponding spans used
            // until s has been computed.
            osipov.slice_container(size);

            // Creates all tuples for this iteration and contains the number of
            // tuples in s
            s = osipov.create_tuples(size, h);
            // Skip all operations till size gets its new size, if this
            // iteration contains no tuples
            if (s > 0) {
                // Wrapper for slicing tuple/aux
                osipov.slice_container(s);

                osipov.stable_sort();

                // Update ranks in tuples and adjust values to sa/isa
                osipov.slice_sa(s);
                osipov.update_ranks();
                osipov.update_container(s);
                osipov.mark_singletons();
            }
            // Adjust to new size.
            size = s;
            // Double considered prefix.
            h = 2 * h;
        }
        phase.split("Write out SA");
        // Write final sa to out_sa.
        osipov.finalize(out_sa);
    }
};
} // Namespace end
