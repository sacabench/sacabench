#pragma once

struct Compare_four_chars;

extern "C"
Compare_four_chars get_new_cmp_four(uint32_t* text);

extern "C"
Compare_four_chars get_new_cmp_four(uint64_t* text);

extern "C"
void set_flags(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux);

extern "C"
void set_flags(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux);

extern "C"
void mark_groups(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux);

extern "C"
void mark_groups(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux);

extern "C"
void initialize_sa(size_t size, uint32_t* sa);

extern "C"
void initialize_sa(size_t size, uint64_t* sa);

extern "C"
void prefix_sum_cub_inclusive(uint32_t* array, Max_without_branching max,
            size_t size);

extern "C"
void prefix_sum_cub_inclusive(uint64_t* array, Max_without_branching max,
            size_t size);

extern "C"
void fill_aux_for_isa(uint32_t* sa, uint32_t* isa, size_t size,
            Compare_four_chars<uint32_t> cmp);

extern "C"
void fill_aux_for_isa(uint64_t* sa, uint64_t* isa, size_t size,
            Compare_four_chars<uint64_t> cmp);

extern "C"
void scatter_to_isa(uint32_t* isa, uint32_t* aux, uint32_t* sa, size_t size);

extern "C"
void scatter_to_isa(uint64_t* isa, uint64_t* aux, uint64_t* sa, size_t size);

extern "C"
void update_ranks_build_aux(uint32_t* h_ranks, uint32_t* aux, size_t size);

extern "C"
void update_ranks_build_aux(uint64_t* h_ranks, uint64_t* aux, size_t size);

extern "C"
void update_ranks_build_aux_tilde(uint32_t* h_ranks, uint32_t* two_h_ranks,
        uint32_t* aux, size_t size);

extern "C"
void update_ranks_build_aux_tilde(uint64_t* h_ranks, uint64_t* two_h_ranks,
            uint64_t* aux, size_t size);

extern "C"
void set_tuple(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
            uint32_t* aux);

extern "C"
void set_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux);

extern "C"
void new_tuple(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
            uint32_t* aux, uint32_t* tuple_index, uint32_t* h_rank);

extern "C"
void new_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux, uint64_t* tuple_index, uint64_t* h_rank);

extern "C"
void isa_to_sa(uint32_t* isa, uint32_t* sa, size_t size);

extern "C"
void isa_to_sa(uint64_t* isa, uint64_t* sa, size_t size);

extern "C"
void generate_two_h_rank(size_t size, size_t h, uint32_t* sa,
            uint32_t* isa, uint32_t* two_h_rank);

extern "C"
void generate_two_h_rank(size_t size, size_t h, uint64_t* sa,
            uint64_t* isa, uint64_t* two_h_rank);
