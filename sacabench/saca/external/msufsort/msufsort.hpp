#include <MSufSort.h>
#include "../external_saca.hpp"

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
   external_saca_with_writable_text<sa_index, uint32_t>(text, out_sa, text.size(), msufsort_ref);
    }

private:
    static void msufsort_ref(unsigned char* text, unsigned int* sa, unsigned int n) {
        MSufSort* m_suffixSorter = new MSufSort;
        m_suffixSorter->Sort(text, n);

        //calculate SA from ISA
        for (int32_t i = 0; i < n; ++i) {
            sa[m_suffixSorter->ISA(i)-1] = i;
        }
    }
};
} // namespace sacabench:reference_sacas
