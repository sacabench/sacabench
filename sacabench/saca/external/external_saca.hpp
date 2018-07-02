/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {

/// \brief Use this if your SACA overwrites the input texts or sentinels,
///        but uses int32 as the SA index type.
template <typename sa_index, typename Fn>
inline void external_saca_with_writable_text(util::string_span text,
                                             util::span<sa_index> out_sa,
                                             size_t n, Fn saca_fn) {

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<int32_t>(n);
    util::container<uint8_t> writeable_text(text);

    if (n < 2) {
        return;
    }
    tdc::StatPhase::resume_tracking();

    { saca_fn(writeable_text.data(), sa_correct_size.data(), n); }

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
inline void external_saca(util::string_span text, util::span<sa_index> out_sa,
                          size_t n, Fn saca_fn) {

    tdc::StatPhase::pause_tracking();
    auto sa_correct_size = util::make_container<int32_t>(n);
    if (n < 2) {
        return;
    }
    tdc::StatPhase::resume_tracking();

    { saca_fn(text.data(), sa_correct_size.data(), n); }

    tdc::StatPhase::pause_tracking();
    for (size_t i = 0; i < n; ++i) {
        out_sa[i] = sa_correct_size[i];
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

        if (n < 2) {
            return;
        }
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

} // namespace sacabench::reference_sacas
