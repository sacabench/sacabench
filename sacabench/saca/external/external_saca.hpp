/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <util/alphabet.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {

/// \brief Use this if your SACA overwrites the input texts or sentinels,
///        but uses int32 as the SA index type.
template <typename sa_index, typename inner_sa_index, typename Fn>
inline void external_saca_with_writable_text(util::string_span text,
                                             util::span<sa_index> out_sa,
                                             size_t n, Fn saca_fn) {

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<inner_sa_index>(n);
    util::container<uint8_t> writeable_text(text);
    tdc::StatPhase::resume_tracking();

    if (n < 2) {
        return;
    }

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
template <typename sa_index, typename inner_sa_type, typename Fn>
inline void sadslike(util::string_span text, util::span<sa_index> out_sa,
                     size_t n, size_t alphabet_size, Fn saca_fn) {

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<inner_sa_type>(n);
    util::container<uint8_t> writeable_text(text);
    tdc::StatPhase::resume_tracking();

    if (n < 2) {
        return;
    }

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
                                     util::span<sa_index> out_sa,
                                     size_t n, Fn saca_fn) {

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<int32_t>(n);
    tdc::StatPhase::resume_tracking();

    if (n < 2) {
        return;
    }

    { saca_fn(text.data(), sa_correct_size.data(), n); }

    tdc::StatPhase::pause_tracking();
    for (size_t i = 0; i < n; ++i) {
        out_sa[i] = sa_correct_size[i];
    }
    tdc::StatPhase::resume_tracking();
}

/// \brief Use this if your SACA doesn't overwrite the input texts or sentinels.
template <typename sa_index, typename Fn32, typename Fn64>
inline void external_saca(util::string_span text,
                          util::span<sa_index> out_sa,
                          size_t n,
                          Fn32 saca_fn_32,
                          Fn64 saca_fn_64) {
    // surpress unused warning
    (void) saca_fn_32;
    (void) saca_fn_64;
    if constexpr (sizeof(sa_index) == 64) {
        if (n < 2) {
            return;
        }
        saca_fn_64(text.data(), out_sa.data(), n);
    } else if constexpr (sizeof(sa_index) == 32) {
        if (n < 2) {
            return;
        }
        saca_fn_32(text.data(), out_sa.data(), n);
    } else {
        tdc::StatPhase::pause_tracking();
        auto sa_correct_size = util::make_container<int64_t>(n);
        tdc::StatPhase::resume_tracking();

        if (n < 2) {
            return;
        }

        {
            saca_fn_64(text.data(), sa_correct_size.data(), n);
        }

        tdc::StatPhase::pause_tracking();
        for (size_t i = 0; i < n; ++i) {
            out_sa[i] = sa_correct_size[i];
        }
        tdc::StatPhase::resume_tracking();
    }
}

/// \brief Use this if your SACA doesn't overwrite the input texts or sentinels,
///        but uses int32 as the SA index type.
template <typename sa_index, typename Fn>
inline void saislike(util::string_span text, util::span<sa_index> out_sa,
                     size_t n, size_t alphabet_size, Fn saca_fn) {

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<int32_t>(n);
    tdc::StatPhase::resume_tracking();

    if (n < 2) {
        return;
    }

    {
        saca_fn(text.data(), sa_correct_size.data(), n, alphabet_size,
                sizeof(util::character), 0);
    }

    tdc::StatPhase::pause_tracking();
    for (size_t i = 0; i < n; ++i) {
        out_sa[i] = sa_correct_size[i];
    }
    tdc::StatPhase::resume_tracking();
}
} // namespace sacabench::reference_sacas
