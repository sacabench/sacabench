#pragma once

#include <iostream>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>


namespace sacabench::div_suf_sort {
template <typename sa_index>
struct utils {
        static constexpr sa_index NEGATIVE_MASK = size_t(1)
                                                  << (sizeof(sa_index) * 8 - 1);
};

    template <typename sa_index>
    struct sa_types {
        enum class s_type { l, s, rms };

        inline static bool is_l_type(sa_index suffix,
                                     util::span<bool> suffix_types) {
            DCHECK_LT(suffix, suffix_types.size());
            return suffix_types[suffix] == 1;
        }

        inline static bool is_s_type(sa_index suffix,
                                     util::span<bool> suffix_types) {
            // Last suffix must be l-type
            DCHECK_LT(suffix, suffix_types.size() - 1);
            return suffix_types[suffix] == 0 && suffix_types[suffix + 1] == 0;
        }

        inline static bool is_rms_type(sa_index suffix,
                                       util::span<bool> suffix_types) {
            // Last suffix must be l-type
            DCHECK_LT(suffix, suffix_types.size() - 1);
            // Checks wether suffix at position suffix is s type and suffix at
            // pos suffix + 1 is l type (i.e. rms)
            return suffix_types[suffix] == 0 && suffix_types[suffix + 1] == 1;
        }
    };

    template <typename sa_index>
    struct rms_suffixes {
        const util::string_span text;
        util::span<sa_index> relative_indices;
        util::span<sa_index> absolute_indices;
        // Consider wether partial_isa is suited for this struct
        util::span<sa_index> partial_isa;
    };

    template <typename sa_index>
    struct buckets {
        size_t alphabet_size;

        // l_buckets containing buckets for l-suffixes of size of alphabet
        util::container<sa_index>& l_buckets;
        // s_buckets containing buckets for s- and rms-suffixes of size
        // of alphabet squared
        util::container<sa_index>& s_buckets;

        // TODO: Check wether size_t for first_letter/second_letter of better use
        inline size_t get_s_bucket_index(size_t first_letter,
                                         size_t second_letter) {
            return first_letter * alphabet_size + second_letter;
        }

        inline size_t get_rms_bucket_index(size_t first_letter,
                                           size_t second_letter) {
            return second_letter * alphabet_size + first_letter;
        }
    };
}
