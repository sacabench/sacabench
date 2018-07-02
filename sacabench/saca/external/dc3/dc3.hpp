/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../external_saca.hpp"
#include "../../../../external/reference_impls/dc3/drittel.C"
#include <cstdint>
#include <util/alphabet.hpp>

namespace sacabench::reference_sacas {

    class dc3 {
    public:
        static constexpr size_t EXTRA_SENTINELS = 3;
        static constexpr char const* NAME = "DC3";
        static constexpr char const* DESCRIPTION = "Difference Cover Modulo 3 SACA";

        template <typename sa_index>
        inline static void construct_sa(util::string_span text_with_sentinels,
                                        const util::alphabet& alphabet,
                                        util::span<sa_index> out_sa) {

            tdc::StatPhase::pause_tracking();

            int n = text_with_sentinels.size();
            std::cout << "n: " << n << std::endl;
            if (n < 4) {
                return;
            }

            auto s = std::make_unique<int[]>(n);
            auto SA = std::make_unique<int[]>(n);
            int K = alphabet.size_with_sentinel();
            std::cout << "K: " << K << std::endl;

            for (size_t index = 0; index < n; index++) {
                s[index] = static_cast<int>(text_with_sentinels[index]);
            }

            tdc::StatPhase::resume_tracking();

            std::cout << " Started ref " << std::endl;
            ::suffixArray(s.get(), SA.get(), n, K);
            std::cout << " Finished ref " << std::endl;

            tdc::StatPhase::pause_tracking();

            for (size_t index = 0; index < n; index++) {
                out_sa[index] = static_cast<sa_index>(SA[index]);
                std::cout << out_sa[index] << ", ";
            }
            std::cout << std::endl;
            tdc::StatPhase::resume_tracking();
        }

    }; // class dc3
} // namespace sacabench::reference_sacas
