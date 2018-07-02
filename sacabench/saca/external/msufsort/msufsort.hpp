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
        tdc::StatPhase::pause_tracking();
        auto sa_correct_size = util::make_container<unsigned int>(out_sa.size());
        util::container<uint8_t> writeable_text(text);

        if (text.size() < 2) {
            return;
        }
        tdc::StatPhase::resume_tracking();

        msufsort_ref(writeable_text.data(), sa_correct_size.data(), out_sa.size());

        tdc::StatPhase::pause_tracking();

        for (size_t i = 0; i < out_sa.size(); ++i) {
            out_sa[i] = sa_correct_size[i];
        }
        tdc::StatPhase::resume_tracking();
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
