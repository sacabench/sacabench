#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <saca_k_reference.hpp>
#pragma GCC diagnostic pop

#include "../external_saca.hpp"

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class saca_k {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "SACA-K_ref";
    static constexpr char const* DESCRIPTION =
        "Reference Implementation of SACA-K by G. Nong.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        sadslike_one_size_only<sa_index, unsigned int>(text, out_sa, text.size(),
                                     alphabet.size_with_sentinel(),
                                     reference_sacas::saca_k_reference::SACA_K);
    }
};
} // namespace sacabench::reference_sacas
