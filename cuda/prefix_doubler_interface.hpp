/*******************************************************************************
 * Copyright (C) 2019 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

struct Max_without_branching;

void word_packing(const uint8_t* chars, uint32_t* result, size_t n);
void word_packing(const uint8_t* chars, uint64_t* result, size_t n);

void set_flags(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux);
void set_flags(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux);

void mark_groups(size_t size, uint32_t* sa, uint32_t* isa, uint32_t* aux);
void mark_groups(size_t size, uint64_t* sa, uint64_t* isa, uint64_t* aux);

void initialize_sa_gpu(size_t size, uint32_t* sa);
void initialize_sa_gpu(size_t size, uint64_t* sa);

void fill_aux_for_isa(uint32_t* text, uint32_t* sa, uint32_t* isa, size_t size);
void fill_aux_for_isa(uint64_t* text, uint64_t* sa, uint64_t* isa, size_t size);

void scatter_to_isa(uint32_t* isa, uint32_t* aux, uint32_t* sa, size_t size);
void scatter_to_isa(uint64_t* isa, uint64_t* aux, uint64_t* sa, size_t size);

void update_ranks_build_aux(uint32_t* h_ranks, uint32_t* aux, size_t size);
void update_ranks_build_aux(uint64_t* h_ranks, uint64_t* aux, size_t size);

void update_ranks_build_aux_tilde(uint32_t* h_ranks, uint32_t* two_h_ranks,
        uint32_t* aux, size_t size);
void update_ranks_build_aux_tilde(uint64_t* h_ranks, uint64_t* two_h_ranks,
            uint64_t* aux, size_t size);

void set_tuple(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
            uint32_t* aux);
void set_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux);

void new_tuple(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
            uint32_t* aux, uint32_t* tuple_index, uint32_t* h_rank);
void new_tuple(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
            uint64_t* aux, uint64_t* tuple_index, uint64_t* h_rank);

void isa_to_sa(uint32_t* isa, uint32_t* sa, size_t size);
void isa_to_sa(uint64_t* isa, uint64_t* sa, size_t size);

void generate_two_h_rank(size_t size, size_t h, uint32_t* sa,
            uint32_t* isa, uint32_t* two_h_rank);
void generate_two_h_rank(size_t size, size_t h, uint64_t* sa,
            uint64_t* isa, uint64_t* two_h_rank);
