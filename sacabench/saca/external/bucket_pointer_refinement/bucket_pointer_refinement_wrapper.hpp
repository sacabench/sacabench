/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <math.h>

#include <util/alphabet.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

#include <kbs_SuffixArrayConstDStepAndPre.h>

namespace sacabench::bucket_pointer_refinement_ext {

class bucket_pointer_refinement_ext {
public:
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "ext_BPR";
    static constexpr char const* DESCRIPTION =
        "Reference implementation of Klaus-Bernd Schürmann";

    template <typename sa_index>
    static void construct_sa(util::string_span input,
                             util::alphabet const& alphabet,
                             util::span<sa_index> sa) {
        tdc::StatPhase bpr("Prepare input file");
        // TODO: Write input to file

        Kbs_Ustring* ustr = NULL;
        Kbs_SuffixArray *sa_ref = NULL;
        Kbs_Ulong q = 3;

        ustr = kbs_getUstring_FromFile(argv[argc-1]);

        bpr.split("Reference Implementation");

        kbs_get_AlphabetForUstring(ustr);
        if (ustr == NULL) {
            printf("unable to get file %s\n", argv[argc-1]);
            KBS_ERROR(KBS_ERROR_NULLPOINTER);
            exit(1);
        }
        if (argc == 2) {
            if (ustr->alphabet->alphaSize <= 9) {
                q = 7;
            }
            if (9 < ustr->alphabet->alphaSize && ustr->alphabet->alphaSize <= 13) {
                q = 6;
            }
            if (13 < ustr->alphabet->alphaSize && ustr->alphabet->alphaSize <= 21) {
                q = 5;
            }
            if (13 < ustr->alphabet->alphaSize && ustr->alphabet->alphaSize <= 21) {
                q = 5;
            }
            if (21 < ustr->alphabet->alphaSize && ustr->alphabet->alphaSize <= 46) {
                q = 4;
            }
            if (46 < ustr->alphabet->alphaSize) {
                q = 3;
            }
        }
        /* implementation using direct pointers as bucket pointers */
        sa_ref = kbs_buildDstepUsePrePlusCopyFreqOrder_SuffixArray(ustr, q);

        bpr.split("Move output");

        // TODO: output sa_ref to sa
        Kbs_Ulong i;
        for(i=0; i<sa->str->strLength; ++i) {
            sa[static_cast<size_t>(i)] = static_cast<sa_index>(sa->posArray[i]);
        }

        kbs_delete_SA_IncludingString(sa_ref);
    }
}; // class bucket_pointer_refinement

} // namespace sacabench::bucket_pointer_refinement_ext
