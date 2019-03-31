#include <../external/reference_impls/sais_lite_reference.hpp>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include "../external_saca.hpp"

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class sais_lite {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "SAIS-LITE_ref";
    static constexpr char const* DESCRIPTION =
        "Suffix Array Induced Sorting by Yuta Mori";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             __attribute__((unused))sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        auto saca_fn = [](auto text_ptr, auto sa_ptr, size_t n) {
            tdc::StatPhase sais_lite("Main Phase");
            sacabench::reference_sacas::sais_lite_reference::sais(text_ptr, sa_ptr, n);
        };

        external_saca_one_size_only<sa_index, int>(
            text, out_sa, text.size(), saca_fn);
    }
};
} // namespace sacabench:reference_sacas
