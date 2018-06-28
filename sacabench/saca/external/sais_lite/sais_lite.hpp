#include <saca/external/sais_lite/sais_lite_reference.hpp>
#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class sais_lite {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Reference_SAIS_LITE";
    static constexpr char const* DESCRIPTION =
        "Suffix Array Induced Sorting by Yuta Mori";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             __attribute__((unused))sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        int* SA = new int[text.size()];

        if (text.size() > 1) {
            tdc::StatPhase sais_lite("Main Phase");
            sacabench::reference_sacas::sais_lite_reference::sais(text.data(), SA, text.size());
        }
        for (size_t i = 0; i < text.size(); i++) {
            DCHECK_LE(static_cast<size_t>(SA[i]), std::numeric_limits<sa_index>::max());
            out_sa[i] = static_cast<sa_index>(SA[i]);
        }
    }
};
} // namespace sacabench:reference_sacas
