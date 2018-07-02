#include "../../../../external/reference_impls/saca_k_reference.hpp"
#include "../external_saca.hpp"
#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>


namespace sacabench::reference_sacas {
using namespace sacabench::util;
class saca_k {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Reference-SACA-K";
    static constexpr char const* DESCRIPTION =
        "Reference Implementation of SACA-K by G. Nong.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        tdc::StatPhase::pause_tracking();
        auto sa_correct_size = util::make_container<unsigned int>(text.size());
        auto writeable_text = util::make_container<uint8_t>(text.size());
        for (size_t i = 0; i < text.size(); ++i) {
            writeable_text[i] = text[i];
        }

        if (text.size() < 2) {
            return;
        }
        tdc::StatPhase::resume_tracking();

        reference_sacas::saca_k_reference::SACA_K(writeable_text.data(), 
            sa_correct_size.data(), text.size(), 
            alphabet.size_with_sentinel(), text.size(), 0);

        tdc::StatPhase::pause_tracking();

        for (size_t i = 0; i < out_sa.size(); ++i) {
            out_sa[i] = sa_correct_size[i];
        }
        tdc::StatPhase::resume_tracking();
    }
};
} // namespace sacabench:reference_sacas
