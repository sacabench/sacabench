#include "../external_saca.hpp"
#pragma GCC diagnostic push
#pragma GCC system_header
#include <MSufSort.h>
#pragma GCC diagnostic pop

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class msufsort {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "MSufSort_ref";
    static constexpr char const* DESCRIPTION =
        "Reference MSufSort by M. Maniscalco.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             const sacabench::util::alphabet&,
                             util::span<sa_index> out_sa) {
        external_saca_with_writable_text_one_size_only<sa_index, int32_t, unsigned char>(
            text, out_sa, text.size(), msufsort_ref);
    }

private:
    static void msufsort_ref(unsigned char* text, int32_t* sa, size_t n) {
        MSufSort m_suffixSorter;
        m_suffixSorter.Sort(text, n);

        // calculate SA from ISA
        for (size_t i = 0; i < n; ++i) {
            sa[m_suffixSorter.ISA(i) - 1] = i;
        }
    }
};
} // namespace sacabench::reference_sacas
