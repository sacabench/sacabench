#include "../../../../external/reference_impls/sads_reference.hpp"
#include "../external_saca.hpp"

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class sads {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Reference_SADS";
    static constexpr char const* DESCRIPTION =
        "Suffix Array D-Critical Sorting by Nong, Zhang and Chan";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        sadslike<sa_index, int32_t>(
            text, out_sa, text.size(), alphabet.size_with_sentinel(),
            sacabench::reference_sacas::sads_reference::SA_DS);
    }
};
} // namespace sacabench::reference_sacas
