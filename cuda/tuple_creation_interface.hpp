#pragma once

struct Compare_first_char;

size_t create_tuples(size_t size, size_t h, uint64_t* sa, uint64_t* isa,
        uint64_t* aux, uint64_t* tuple_index, uint64_t* h_rank);

size_t create_tuples(size_t size, size_t h, uint32_t* sa, uint32_t* isa,
        uint32_t* aux, uint32_t* tuple_index, uint32_t* h_rank);

void initialize(size_t n, const char* text, uint32_t* sa, uint32_t* isa,
            uint32_t* aux);

void initialize(size_t n, const char* text, uint64_t* sa, uint64_t* isa,
            uint64_t* aux);
