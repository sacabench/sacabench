#include "../../../../external/reference_impls/saca-k_reference.hpp"
#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class saca-k {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Reference_SACA-K";
    static constexpr char const* DESCRIPTION =
        "Reference Implementation of SACA-K by G. Nong.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        //TODO: Call SACA-K
    }
};
} // namespace sacabench:reference_sacas
