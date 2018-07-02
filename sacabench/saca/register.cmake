# Syntax: SACA_REGISTER(<path to header> <c++ type>)

# Reference SACAS:
SACA_REGISTER("saca/external/qsufsort/qsufsort_wrapper.hpp"
    sacabench::qsufsort_ext::qsufsort_ext)

SACA_REGISTER("saca/external/deep_shallow.hpp"
    sacabench::reference_sacas::deep_shallow)

SACA_REGISTER("saca/external/divsufsort.hpp"
    sacabench::reference_sacas::div_suf_sort)

SACA_REGISTER("saca/external/gsaca.hpp"
        sacabench::reference_sacas::gsaca)

# Our implementations:

SACA_REGISTER("saca/deep_shallow/saca.hpp"
    sacabench::deep_shallow::saca)

SACA_REGISTER("saca/deep_shallow/blind/sort.hpp"
    sacabench::deep_shallow::blind::saca)

SACA_REGISTER("saca/bucket_pointer_refinement.hpp"
    sacabench::bucket_pointer_refinement::bucket_pointer_refinement)

SACA_REGISTER("saca/m_suf_sort.hpp"
    sacabench::m_suf_sort::m_suf_sort2)

SACA_REGISTER("saca/prefix_doubling.hpp"
    sacabench::prefix_doubling::prefix_doubling)

SACA_REGISTER("saca/prefix_doubling.hpp"
    sacabench::prefix_doubling::prefix_doubling_discarding)

SACA_REGISTER("saca/sais.hpp"
    sacabench::sais::sais)

SACA_REGISTER("saca/external/sais/sais.hpp"
    sacabench::reference_sacas::sais)

SACA_REGISTER("saca/external/sais_lite/sais_lite.hpp"
    sacabench::reference_sacas::sais_lite)    

SACA_REGISTER("saca/sads.hpp"
    sacabench::sads::sads)

SACA_REGISTER("saca/external/sads/sads.hpp"
    sacabench::reference_sacas::sads)

SACA_REGISTER("saca/gsaca.hpp"
    sacabench::gsaca::gsaca)

SACA_REGISTER("saca/dc7.hpp"
    sacabench::dc7::dc7)

SACA_REGISTER("saca/qsufsort.hpp"
    sacabench::qsufsort::qsufsort)

SACA_REGISTER("saca/naive.hpp"
    sacabench::naive::naive)

SACA_REGISTER("saca/sacak.hpp"
    sacabench::sacak::sacak)

SACA_REGISTER("saca/dc3.hpp"
    sacabench::dc3::dc3)

SACA_REGISTER("saca/nzSufSort.hpp"
    sacabench::nzsufsort::nzsufsort)

SACA_REGISTER("saca/dc3_lite.hpp"
    sacabench::dc3_lite::dc3_lite)
