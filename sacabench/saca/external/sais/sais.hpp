#include <saca/external/sais/sais_reference.hpp>
#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class sais {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Reference_SAIS";
    static constexpr char const* DESCRIPTION =
        "Suffix Array Induced Sorting by Nong, Zhang and Chan";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        int* SA = new int[text.size()];

        if (text.size() > 1) {
            tdc::StatPhase sais("Main Phase");
            sacabench::reference_sacas::sais_reference::SAIS(text.data(), SA, text.size(), alphabet.max_character_value() + 1, sizeof(char), 0);
        }
        for (size_t i = 0; i < text.size(); i++) {
            DCHECK_LE(static_cast<size_t>(SA[i]), std::numeric_limits<sa_index>::max());
            out_sa[i] = static_cast<sa_index>(SA[i]);
        }
    }
};
} // namespace sacabench:reference_sacas
