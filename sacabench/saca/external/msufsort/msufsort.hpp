/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <MSufSort.h>
#include "../external_saca.hpp"
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class msufsort {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Reference-MSufSort";
    static constexpr char const* DESCRIPTION =
        "Reference MSufSort by M. Maniscalco.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        external_saca_with_writable_text<sa_index>(text, out_sa, text.size(), msufsort_ref);
    }
    
private:
    static void msufsort_ref(unsigned char* text, int32_t* sa, int32_t n) {
        MSufSort* m_suffixSorter = new MSufSort;
        m_suffixSorter->Sort(text, n);
        
        for (int32_t i = 0; i < n; ++i) {
            std::cout << m_suffixSorter->ISA(i)-1 << ", ";
        }
        std::cout << std::endl;
        
        //calculate SA from ISA
        for (int32_t i = 0; i < n; ++i) {
            sa[m_suffixSorter->ISA(i)-1] = i;
        }
        
        for (int32_t i = 0; i < n; ++i) {
            std::cout << sa[i] << ", ";
        }
        std::cout << std::endl;
    }
};
} // namespace sacabench:reference_sacas
