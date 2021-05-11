#pragma once

#include "external_saca.hpp"
#include <util/span.hpp>
#include <cstdint>

extern "C" int32_t libsais_omp(const uint8_t * T, int32_t * SA, int32_t n, int32_t fs, int32_t threads);
extern "C" int64_t libsais64_omp(const uint8_t * T, int64_t * SA, int64_t n, int64_t fs, int64_t threads);

namespace sacabench::reference_sacas 
{
    class libsais_seq 
    {
    public:
        static constexpr size_t EXTRA_SENTINELS = 0;
        static constexpr char const *NAME = "libsais_seq_ref";
        static constexpr char const *DESCRIPTION = "Computes a suffix array with the sequential algorithm libsais by Ilya Grebnov.";

        inline static int32_t libsais32_seq(const uint8_t * T, int32_t * SA, int32_t n)
        {
            return libsais_omp(T, SA, n, 0, 1);
        }

        inline static int64_t libsais64_seq(const uint8_t * T, int64_t * SA, int64_t n)
        {
            return libsais64_omp(T, SA, n, 0, 1);
        }

        template <typename sa_index>
        inline static void construct_sa(util::string_span text, const util::alphabet&, util::span<sa_index> out_sa)
        {
            external_saca<sa_index>(text, out_sa, text.size(), libsais32_seq, libsais64_seq);
        }
    };

    class libsais_par 
    {
    public:
        static constexpr size_t EXTRA_SENTINELS = 0;
        static constexpr char const *NAME = "libsais_par_ref";
        static constexpr char const *DESCRIPTION = "Computes a suffix array with the parallel algorithm libsais by Ilya Grebnov.";

        inline static int32_t libsais32_par(const uint8_t * T, int32_t * SA, int32_t n)
        {
            return libsais_omp(T, SA, n, 0, 0);
        }

        inline static int64_t libsais64_par(const uint8_t * T, int64_t * SA, int64_t n)
        {
            return libsais64_omp(T, SA, n, 0, 0);
        }

        template <typename sa_index>
        inline static void construct_sa(util::string_span text, const util::alphabet&, util::span<sa_index> out_sa)
        {
            external_saca<sa_index>(text, out_sa, text.size(), libsais32_par, libsais64_par);
        }
    };
}
