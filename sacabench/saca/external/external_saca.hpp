/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <tudocomp_stat/StatPhase.hpp>
#include <util/alphabet.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::reference_sacas {

/// \brief Allocates a untracked container<inner_sa_index>, calls the algorithm
/// with it, and copies the result out of it.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void with_sa_copy(util::string_span text, util::span<sa_index> out_sa,
                         size_t n, Fn saca_fn) {
    DCHECK(out_sa.size() == n);
    DCHECK(text.size() >= n);

    util::container<inner_sa_index> sa_correct_size;
    {
        tdc::StatPhase::pause_tracking();
        sa_correct_size = util::make_container<inner_sa_index>(n);
        tdc::StatPhase::resume_tracking();
    }
    { saca_fn(text.data(), sa_correct_size.data(), n); }
    {
        tdc::StatPhase::pause_tracking();
        for (size_t i = 0; i < n; ++i) {
            out_sa[i] = sa_correct_size[i];
        }
        { auto dropme1 = std::move(sa_correct_size); }
        tdc::StatPhase::resume_tracking();
    }
}

/// \brief Allocates a untracked container<inner_sa_index> and a copy of the
/// input, calls the algorithm with it, and copies the result out of it.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void with_sa_and_text_copy(util::string_span text,
                                  util::span<sa_index> out_sa, size_t n,
                                  Fn saca_fn) {
    DCHECK(out_sa.size() == n);
    DCHECK(text.size() >= n);

    util::container<inner_sa_index> sa_correct_size;
    util::container<util::character> writeable_text;
    {
        tdc::StatPhase::pause_tracking();
        util::allow_container_copy _guard;
        sa_correct_size = util::make_container<inner_sa_index>(n);
        writeable_text = text;
        tdc::StatPhase::resume_tracking();
    }
    { saca_fn(writeable_text.data(), sa_correct_size.data(), n); }
    {
        tdc::StatPhase::pause_tracking();
        for (size_t i = 0; i < n; ++i) {
            out_sa[i] = sa_correct_size[i];
        }
        {
            auto dropme1 = std::move(sa_correct_size);
            auto dropme2 = std::move(writeable_text);
        }
        tdc::StatPhase::resume_tracking();
    }
}

/// \brief Allocates a copy of the
/// input, calls the algorithm with it, and copies the result out of it.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void with_text_copy(util::string_span text, util::span<sa_index> out_sa,
                           size_t n, Fn saca_fn) {
    DCHECK(out_sa.size() == n);
    DCHECK(text.size() >= n);

    util::container<util::character> writeable_text;
    {
        tdc::StatPhase::pause_tracking();
        util::allow_container_copy _guard;
        writeable_text = text;
        tdc::StatPhase::resume_tracking();
    }
    { saca_fn(writeable_text.data(), (inner_sa_index*)out_sa.data(), n); }
    {
        tdc::StatPhase::pause_tracking();
        { auto dropme2 = std::move(writeable_text); }
        tdc::StatPhase::resume_tracking();
    }
}

/// \brief Errors out if the input is too long and does an early return if it is
/// too short.
template <typename check_index>
inline bool early_check(size_t n) {
    if (std::numeric_limits<check_index>::max() < n) {
        std::cerr
            << "ERROR: This algorithm only supports inputs addressable with "
            << (sizeof(check_index) * CHAR_BIT) << "Bit" << std::endl;
        return true;
    }
    if (n < 2) {
        return true;
    }
    return false;
}

/// \brief Changes the SA span such that it starts at the right position
/// for a input with extra sentinel bytes.
template <typename sa_index>
inline void adjust_sa_span_for_sentinels(util::span<sa_index>* out_sa,
                                         util::string_span text, size_t n) {
    DCHECK(out_sa->size() == text.size());
    DCHECK(out_sa->size() >= n);
    DCHECK(text.size() >= n);
    size_t sentinels = text.size() - n;

    *out_sa = out_sa->slice(sentinels, n + sentinels);
}

template <typename sa_index>
constexpr size_t INDEX_BITS = sizeof(sa_index) * CHAR_BIT;

//////// main interface

/// \brief Use this if your SACA doesn't overwrite the input texts or sentinels,
///        and uses both int32_t and int64_t as SA index_type.
template <typename sa_index, typename Fn32, typename Fn64>
inline void external_saca(util::string_span text, util::span<sa_index> out_sa,
                          size_t n, Fn32 saca_fn_32, Fn64 saca_fn_64) {
    if (early_check<int64_t>(n)) {
        return;
    }
    adjust_sa_span_for_sentinels(&out_sa, text, n);

    // surpress unused warning
    (void)saca_fn_32;
    (void)saca_fn_64;
    if constexpr (INDEX_BITS<sa_index> == 32) {
        saca_fn_32(text.data(), (int32_t*)out_sa.data(), n);
    } else if constexpr (INDEX_BITS<sa_index> == 64) {
        saca_fn_64(text.data(), (int64_t*)out_sa.data(), n);
    } else {
        with_sa_copy<sa_index, int64_t>(text, out_sa, n, saca_fn_64);
    }
}

/// \brief Use this if your SACA doesn't overwrite the input texts or sentinels,
///        but uses only a single SA index_type.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void external_saca_one_size_only(util::string_span text,
                                        util::span<sa_index> out_sa, size_t n,
                                        Fn saca_fn) {
    if (early_check<inner_sa_index>(n)) {
        return;
    }
    adjust_sa_span_for_sentinels(&out_sa, text, n);

    if constexpr (INDEX_BITS<sa_index> == INDEX_BITS<inner_sa_index>) {
        saca_fn(text.data(), (inner_sa_index*)out_sa.data(), n);
    } else {
        with_sa_copy<sa_index, inner_sa_index>(text, out_sa, n, saca_fn);
    }
}

/// \brief Use this if your SACA overwrites the input texts or sentinels,
///        but uses only a single SA index_type.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void external_saca_with_writable_text_one_size_only(
    util::string_span text, util::span<sa_index> out_sa, size_t n, Fn saca_fn) {
    if (early_check<inner_sa_index>(n)) {
        return;
    }
    adjust_sa_span_for_sentinels(&out_sa, text, n);

    if constexpr (INDEX_BITS<sa_index> == INDEX_BITS<inner_sa_index>) {
        with_text_copy<sa_index, inner_sa_index>(text, out_sa, n, saca_fn);
    } else {
        with_sa_and_text_copy<sa_index, inner_sa_index>(text, out_sa, n,
                                                        saca_fn);
    }
}

template <typename Fn>
inline auto sadslike_adapter(size_t alphabet_size, Fn saca_fn) {
    return [alphabet_size, saca_fn](auto text_ptr, auto sa_ptr, size_t n) {
        saca_fn(text_ptr, sa_ptr, n, alphabet_size, n, 0);
    };
}

template <typename Fn>
inline auto saislike_adapter(size_t alphabet_size, Fn saca_fn) {
    return [alphabet_size, saca_fn](auto text_ptr, auto sa_ptr, size_t n) {
        saca_fn(text_ptr, sa_ptr, n, alphabet_size, sizeof(util::character), 0);
    };
}

} // namespace sacabench::reference_sacas
