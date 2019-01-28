/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>
#include <tuple>

#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/sort/ips4o.hpp>
#include <util/sort/std_sort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::util {
class sa_check_result {
public:
    enum cases {
        ok,
        wrong_length,
        not_a_permutation,
        not_suffix_sorted,
    };

    inline sa_check_result(cases v) : value(v) {}

    inline explicit operator bool() const {
        return value == sa_check_result::ok;
    }

    inline friend bool operator==(sa_check_result const& lhs,
                                  sa_check_result const& rhs) {
        return lhs.value == rhs.value;
    }
    inline friend bool operator!=(sa_check_result const& lhs,
                                  sa_check_result const& rhs) {
        return lhs.value != rhs.value;
    }

    inline friend std::ostream& operator<<(std::ostream& out,
                                           sa_check_result const& x) {
        switch (x.value) {
        case cases::wrong_length:
            return out << "wrong_length";
        case cases::not_a_permutation:
            return out << "not_a_permutation";
        case cases::not_suffix_sorted:
            return out << "not_suffix_sorted";
        default:
            return out << "ok";
        }
    }

private:
    cases value;
};

template <typename sa_index_type, typename sorter_type>
sa_check_result sa_check_naive_sorter(span<sa_index_type> sa, string_span text,
                                      sorter_type sorter) {
    DCHECK(can_represent_all_values<sa_index_type>(sa.size() + 1));

    if (sa.size() != text.size()) {
        return sa_check_result::wrong_length;
    }
    size_t const N = text.size();

    // Create an container of every index positions.
    auto naive = util::make_container<sa_index_type>(N);
    for (size_t i = 0; i < N; i++) {
        naive[i] = i;
    }

    // Construct a SA by sorting according
    // to the suffix starting at that index.
    sorter(naive.slice(),
           util::compare_key([&](size_t i) { return text.slice(i); }));

    for (size_t i = 0; i < N; i++) {
        if (naive[i] != sa[i]) {
            return sa_check_result::not_suffix_sorted;
        }
    }

    return sa_check_result::ok;
}

template <typename sa_index_type>
sa_check_result sa_check_naive(span<sa_index_type> sa, string_span text) {
    return sa_check_naive_sorter<sa_index_type>(
        sa, text, [](auto... args) { sort::std_sort(args...); });
}

template <typename sa_index_type, typename sorter_type>
sa_check_result sa_check_sorter(span<sa_index_type> sa, string_span text,
                                sorter_type sorter) {
    // Check for size + 1 because the algorithm
    // calculates maxvalue + 1 at one point.
    DCHECK(can_represent_all_values<sa_index_type>(sa.size() + 1));

    if (sa.size() != text.size()) {
        return sa_check_result::wrong_length;
    }
    size_t const N = text.size();

    struct pair {
        sa_index_type text_pos;
        sa_index_type sa_pos;
    };
    auto P = make_container<pair>(N);

    for (size_t i = 0; i < N; ++i) {
        P[i] = pair{sa[i], sa_index_type(i + 1)};
    }

    sorter(P.slice(), [](auto const& lhs, auto const& rhs) {
        return lhs.text_pos < rhs.text_pos;
    });

    for (size_t i = 0; i < N; ++i) {
        if (P[i].text_pos != static_cast<sa_index_type>(i)) {
            return sa_check_result::not_a_permutation;
        }
    }

    struct tripple {
        sa_index_type sa_pos;
        character chr;
        sa_index_type sa_pos_2;
    };
    auto S = make_container<tripple>(N);

    for (size_t i = 0; i < N; ++i) {
        sa_index_type r1 = P[i].sa_pos;
        sa_index_type r2;
        if (i + 1 < N) {
            r2 = P[i + 1].sa_pos;
        } else {
            r2 = 0;
        }

        S[i] = tripple{r1, text[i], r2};
    }

    sorter(S.slice(), [](auto const& lhs, auto const& rhs) {
        return lhs.sa_pos < rhs.sa_pos;
    });

    for (size_t i = 1; i < N; ++i) {
        auto const& a = S[i - 1];
        auto const& b = S[i];
        if (std::make_tuple(a.chr, a.sa_pos_2) >
            std::make_tuple(b.chr, b.sa_pos_2)) {
            return sa_check_result::not_suffix_sorted;
        }
    }

    return sa_check_result::ok;
}

template <typename sa_index_type>
sa_check_result sa_check(span<sa_index_type> sa, string_span text) {
    return sa_check_sorter<sa_index_type>(
        sa, text, [](auto... args) { sort::std_sort(args...); });
}

template <typename sa_index_type>
sa_check_result sa_check_dispatch(span<sa_index_type> sa, string_span text,
                                  bool fast) {
    if (fast) {
        return sa_check_sorter(
            sa, text, [](auto... args) { sort::ips4o_sort_parallel(args...); });
    }
    return sa_check(sa, text);
}

} // namespace sacabench::util
