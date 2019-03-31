#include "../../../../external/reference_impls/sais_reference.hpp"
#include <util/alphabet.hpp>
#include <tudocomp_stat/StatPhase.hpp>
#include "../external_saca.hpp"

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class sais {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "SAIS_ref";
    static constexpr char const* DESCRIPTION =
        "Suffix Array Induced Sorting by Nong, Zhang and Chan";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        saislike_one_size_only<sa_index>(text, out_sa, text.size(),
                           alphabet.size_with_sentinel(),
                           sacabench::reference_sacas::sais_reference::SAIS);
    }
};
} // namespace sacabench::reference_sacas
