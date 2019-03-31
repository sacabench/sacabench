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

template <typename sa_index, typename inner_sa_index, typename Fn>
inline void call_with_untracked_inner_sa_index_buffer(
    util::string_span text, util::span<sa_index> out_sa, size_t n, Fn saca_fn) {
    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<inner_sa_index>(n);
    tdc::StatPhase::resume_tracking();

    { saca_fn(text.data(), sa_correct_size.data(), n); }

    tdc::StatPhase::pause_tracking();
    for (size_t i = 0; i < n; ++i) {
        out_sa[i] = sa_correct_size[i];
    }
    tdc::StatPhase::resume_tracking();
}

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

/// \brief Use this if your SACA overwrites the input texts or sentinels,
///        but uses int32 as the SA index type.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void external_saca_with_writable_text(util::string_span text,
                                             util::span<sa_index> out_sa,
                                             size_t n, Fn saca_fn) {
    if (early_check<inner_sa_index>(text.size())) {
        return;
    }

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<inner_sa_index>(n);
    util::container<uint8_t> writeable_text(text);
    tdc::StatPhase::resume_tracking();

    { saca_fn(writeable_text.data(), sa_correct_size.data(), n); }

    tdc::StatPhase::pause_tracking();
    const size_t SENTINELS = text.size() - n;

    for (size_t i = 0; i < n; ++i) {
        out_sa[SENTINELS + i] = sa_correct_size[i];
    }
    tdc::StatPhase::resume_tracking();
}

/// \brief Use this if your SACA overwrites the input texts or sentinels,
///        but uses int32 as the SA index type.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void sadslike(util::string_span text, util::span<sa_index> out_sa,
                     size_t n, size_t alphabet_size, Fn saca_fn) {
    if (early_check<inner_sa_index>(text.size())) {
        return;
    }

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<inner_sa_index>(n);
    util::container<uint8_t> writeable_text(text);
    tdc::StatPhase::resume_tracking();

    {
        saca_fn(writeable_text.data(), sa_correct_size.data(), n, alphabet_size,
                n, 0);
    }

    tdc::StatPhase::pause_tracking();
    const size_t SENTINELS = text.size() - n;

    for (size_t i = 0; i < n; ++i) {
        out_sa[SENTINELS + i] = sa_correct_size[i];
    }
    tdc::StatPhase::resume_tracking();
}

/// \brief Use this if your SACA doesn't overwrite the input texts or sentinels,
///        but uses int32 as the SA index type.
template <typename sa_index, typename Fn>
inline void external_saca_32bit_only(util::string_span text,
                                     util::span<sa_index> out_sa, size_t n,
                                     Fn saca_fn) {
    if (early_check<int32_t>(text.size())) {
        return;
    }

    if constexpr (sizeof(sa_index) == 32) {
        saca_fn(text.data(), out_sa.data(), n);
    } else {
        call_with_untracked_inner_sa_index_buffer<sa_index, int32_t>(
            text, out_sa, n, saca_fn);
    }
}

/// \brief Use this if your SACA doesn't overwrite the input texts or sentinels.
template <typename sa_index, typename Fn32, typename Fn64>
inline void external_saca(util::string_span text, util::span<sa_index> out_sa,
                          size_t n, Fn32 saca_fn_32, Fn64 saca_fn_64) {
    if (early_check<int64_t>(text.size())) {
        return;
    }

    // surpress unused warning
    (void)saca_fn_32;
    (void)saca_fn_64;
    if constexpr (sizeof(sa_index) == 64) {
        saca_fn_64(text.data(), out_sa.data(), n);
    } else if constexpr (sizeof(sa_index) == 32) {
        saca_fn_32(text.data(), out_sa.data(), n);
    } else {
        call_with_untracked_inner_sa_index_buffer<sa_index, int64_t>(
            text, out_sa, n, saca_fn_64);
    }
}

/// \brief Use this if your SACA doesn't overwrite the input texts or sentinels,
///        but uses int32 as the SA index type.
template <typename sa_index, typename Fn>
inline void saislike(util::string_span text, util::span<sa_index> out_sa,
                     size_t n, size_t alphabet_size, Fn saca_fn) {
    if (early_check<int32_t>(text.size())) {
        return;
    }

    call_with_untracked_inner_sa_index_buffer<sa_index, int32_t>(
        text, out_sa, n,
        [alphabet_size, &saca_fn](util::character const* text_ptr,
                                  int32_t* sa_ptr, size_t n) {
            saca_fn(text_ptr, sa_ptr, n, alphabet_size, sizeof(util::character),
                    0);
        });
}
} // namespace sacabench::reference_sacas
