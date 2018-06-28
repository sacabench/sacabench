// TODO: Update Copyright header
/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <math.h>
#include <stdio.h>

#include <util/alphabet.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/sort/bucketsort.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

extern "C" {
#include <kbs_Error.h>
#include <kbs_Math.h>
#include <kbs_String.h>
#include <kbs_SuffixArray.h>
#include <kbs_SuffixArrayAnnotated.h>
#include <kbs_SuffixArrayChecker.h>
#include <kbs_SuffixArrayConstDStepAndPre.h>
#include <kbs_Types.h>
}

namespace sacabench::bucket_pointer_refinement_ext {

class bucket_pointer_refinement_ext {
public:
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "ext_BPR";
    static constexpr char const* DESCRIPTION =
        "Reference implementation of Klaus-Bernd Schürmann";

    template <typename sa_index>
    static void construct_sa(util::string_span input,
                             util::alphabet const& /*alphabet*/,
                             util::span<sa_index> sa_return) {
        tdc::StatPhase bpr("Prepare input file");

        // store contents in temporary file
        char filename_template[] = "/tmp/bpr_tempfile_XXXXXX";
        if (mkstemp(filename_template) == -1) {
            exit(1);
        }
        std::string filename = std::string(filename_template);
        std::ofstream out_file(filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
        out_file << input;
        out_file.flush();
        out_file.close();

        Kbs_Ustring* ustr = NULL;
        Kbs_SuffixArray *sa = NULL;
        Kbs_Ulong q = 3;
        ustr = kbs_getUstring_FromFile(filename.c_str());

        bpr.split("Reference Implementation");

        kbs_get_AlphabetForUstring(ustr);
        if (ustr == NULL) {
            printf("unable to get file %s\n", filename.c_str());
            KBS_ERROR(KBS_ERROR_NULLPOINTER);
            exit(1);
        }

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

        sa = kbs_buildDstepUsePrePlusCopyFreqOrder_SuffixArray(ustr, q);

        bpr.split("Move output");

        Kbs_Ulong i;
        for(i=0; i<sa->str->strLength; ++i) {
            sa_return[static_cast<size_t>(i)] = static_cast<sa_index>(sa->posArray[i]);
        }

        // clean up
        kbs_delete_SA_IncludingString(sa);

        remove(filename.c_str());
    }
}; // class bucket_pointer_refinement

} // namespace sacabench::bucket_pointer_refinement_ext
