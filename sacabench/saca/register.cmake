# Syntax: SACA_REGISTER(<path to header> <c++ type>)

# Reference SACAS:
SACA_REGISTER("saca/external/deep_shallow.hpp"
    sacabench::reference_sacas::deep_shallow)

SACA_REGISTER("saca/external/divsufsort.hpp"
    sacabench::reference_sacas::div_suf_sort)

SACA_REGISTER("saca/external/parallel_divsufsort/parallel_divsufsort.hpp"
    sacabench::reference_sacas::parallel_div_suf_sort)

SACA_REGISTER("saca/external/msufsort/msufsort.hpp"
    sacabench::reference_sacas::msufsort)

SACA_REGISTER("saca/external/saca_k/saca_k.hpp"
    sacabench::reference_sacas::saca_k)

SACA_REGISTER("saca/external/sads/sads.hpp"
    sacabench::reference_sacas::sads)

SACA_REGISTER("saca/external/sais/sais.hpp"
    sacabench::reference_sacas::sais)

SACA_REGISTER("saca/external/sais_lite/sais_lite.hpp"
    sacabench::reference_sacas::sais_lite)

SACA_REGISTER("saca/external/gsaca.hpp"
    sacabench::reference_sacas::gsaca)

SACA_REGISTER("saca/external/qsufsort/qsufsort_wrapper.hpp"
    sacabench::qsufsort_ext::qsufsort_ext)

SACA_REGISTER("saca/external/dc3/dc3.hpp"
        sacabench::reference_sacas::dc3)

# Our implementations:

SACA_REGISTER("saca/deep_shallow/saca.hpp"
    sacabench::deep_shallow::serial)

SACA_REGISTER("saca/deep_shallow/saca.hpp"
    sacabench::deep_shallow::serial_big_buckets)

SACA_REGISTER("saca/deep_shallow/saca.hpp"
    sacabench::deep_shallow::parallel)

SACA_REGISTER("saca/deep_shallow/saca.hpp"
    sacabench::deep_shallow::parallel)

#SACA_REGISTER("saca/deep_shallow/blind/sort.hpp"
#    sacabench::deep_shallow::blind::saca)

SACA_REGISTER("saca/bucket_pointer_refinement.hpp"
    sacabench::bucket_pointer_refinement::bucket_pointer_refinement)

SACA_REGISTER("saca/bucket_pointer_refinement_parallel.hpp"
    sacabench::bucket_pointer_refinement_parallel::bucket_pointer_refinement_parallel)

SACA_REGISTER("saca/external/bucket_pointer_refinement/bucket_pointer_refinement_wrapper.hpp"
    sacabench::bucket_pointer_refinement_ext::bucket_pointer_refinement_ext)

SACA_REGISTER("saca/m_suf_sort.hpp"
    sacabench::m_suf_sort::m_suf_sort2)

SACA_REGISTER("saca/prefix_doubling.hpp"
    sacabench::prefix_doubling::prefix_doubling)

SACA_REGISTER("saca/prefix_doubling.hpp"
    sacabench::prefix_doubling::prefix_discarding_2)

SACA_REGISTER("saca/prefix_doubling.hpp"
    sacabench::prefix_doubling::prefix_discarding_4)

SACA_REGISTER("saca/prefix_doubling.hpp"
    sacabench::prefix_doubling::prefix_discarding_4_parallel)

SACA_REGISTER("saca/sais.hpp"
    sacabench::sais::sais)

SACA_REGISTER("saca/parallel_sais.hpp"
    sacabench::parallel_sais::parallel_sais)

SACA_REGISTER("saca/sads.hpp"
    sacabench::sads::sads)

SACA_REGISTER("saca/gsaca/gsaca.hpp"
    sacabench::gsaca::gsaca)

SACA_REGISTER("saca/gsaca/gsaca_new.hpp"
        sacabench::gsaca::gsaca_new)

SACA_REGISTER("saca/gsaca/gsaca_parallel.hpp"
        sacabench::gsaca::gsaca_parallel)

SACA_REGISTER("saca/dc7.hpp"
    sacabench::dc7::dc7)

SACA_REGISTER("saca/qsufsort.hpp"
    sacabench::qsufsort::qsufsort)

SACA_REGISTER("saca/naive.hpp"
    sacabench::naive::naive)

SACA_REGISTER("saca/naive.hpp"
    sacabench::naive::naive_ips4o)

SACA_REGISTER("saca/naive.hpp"
    sacabench::naive::naive_ips4o_parallel)

SACA_REGISTER("saca/naive.hpp"
    sacabench::naive::naive_parallel)

SACA_REGISTER("saca/sacak.hpp"
    sacabench::sacak::sacak)

SACA_REGISTER("saca/dc3.hpp"
    sacabench::dc3::dc3)

SACA_REGISTER("saca/div_suf_sort/saca.hpp"
    sacabench::div_suf_sort::div_suf_sort)

SACA_REGISTER("saca/nzSufSort.hpp"
    sacabench::nzsufsort::nzsufsort)

SACA_REGISTER("saca/dc3_lite.hpp"
    sacabench::dc3_lite::dc3_lite)

SACA_REGISTER("saca/dc3_par.hpp"
    sacabench::dc3_par::dc3_par)

SACA_REGISTER("saca/dc3_par2.hpp"
    sacabench::dc3_par2::dc3_par2)

SACA_REGISTER("saca/osipov/osipov_sequential.hpp"
    sacabench::osipov::osipov_sequential)

SACA_REGISTER("saca/osipov/osipov_sequential.hpp"
    sacabench::osipov::osipov_sequential_wp)

SACA_REGISTER("saca/osipov/osipov_parallel.hpp"
    sacabench::osipov::osipov_parallel)

SACA_REGISTER("saca/osipov/osipov_parallel.hpp"
    sacabench::osipov::osipov_parallel_wp)

if(SACA_ENABLE_CUDA)
SACA_REGISTER("saca/osipov/osipov_gpu.hpp"
    sacabench::osipov::osipov_gpu)
endif()
