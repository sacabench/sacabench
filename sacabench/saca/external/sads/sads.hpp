#include "../../../../external/reference_impls/sads_reference.hpp"
#include <util/alphabet.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include "../external_saca.hpp"

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class sads {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "SADS_ref";
    static constexpr char const* DESCRIPTION =
        "Suffix Array D-Critical Sorting by Nong, Zhang and Chan";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        auto saca_fn = sadslike_adapter(alphabet.size_with_sentinel(),
                                        sacabench::reference_sacas::sads_reference::SA_DS);
        external_saca_one_size_only<sa_index, int32_t>(
            text, out_sa, text.size(),
            saca_fn);
    }
};
} // namespace sacabench::reference_sacas
