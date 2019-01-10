#pragma once

#include <prefix_doubling_interface.hpp>
#include <sacabench/util/span.hpp>


template <typename sa_index>
struct osipov_gpu {
private:
    sa_index* text;
    //TODO: Wrapper for creation of cmp-function!
    Compare_four_chars<sa_index> cmp;
    // Three base arrays
    sa_index* sa;
    sa_index* isa;
    sa_index* aux;
    size_t size;

    // Arrays for tuples (use sa for tuple_index)
    // "Inducing reference" (third component in tuple)
    sa_index* two_h_rank;
    // Rank at depth h (second component in tuple)
    sa_index* h_rank;

public:

    osipov_gpu(size_t size, sa_index* text, sa_index* sa, sa_index* isa,
            sa_index* aux, sa_index* two_h_rank, sa_index* h_rank) : size(size),
            text(text), sa(sa), isa(isa), aux(aux), two_h_rank(two_h_rank),
            h_rank(h_rank), cmp(get_new_cmp_four(text)) {}

    /*
        Getter for debugging.
    */

    sa_index* get_sa() {return sa;}

    sa_index* get_isa() {return isa;}

    sa_index* get_aux() {return aux;}

    sa_index* get_rank() {return h_rank;}

    sa_index* get_two_rank() {return two_h_rank;}

    size_t get_size() {return size;}

    /*
        \brief slices array, i.e. sets the correct size.
    */
    void slice_container(size_t s) {size = s;}

    /*
        \brief method which is called in osipov.hpp but doesn't actually do
        anything.
        TODO: Check if logic can be moved to slice_container in cpu versions.
    */
    void slice_sa(size_t s) {}

    /*
        \brief PLACEHOLDER FUNCTION: Checks each h-group for singleton groups and
        marks these groups by inverting them.

        TODO: Replace with efficient version (see mark_singletons.cu).
    */
    void mark_singletons() {
        if(size > 0) {


            // Move sa, isa to shared memory

            set_flags(size, sa, isa, aux);
            cudaDeviceSynchronize();
            mark_groups(size, sa, isa, aux);
            cudaDeviceSynchronize();
        }
    }

    /*
        \brief Init SA on GPU. Every GPU thread writes his index size_to SA,
        then jumps stride size until end is reached.

        Wrapper for kernel
    */
    void initialize_sa() {
        initialize_sa_gpu(size, sa);
    }

    /*
        Initially sorts SA according to text using the CUB Radixsort
    */
    void inital_sort() {
        //TODO: similar Wrapper for SortPairs as for prefix_sum_cub
         //Actual values; use h_rank as temp storage
        auto keys_out = h_rank;     // e.g., [        ...        ]


        // Allocate Unified Memory – accessible from CPU or GPU
        // cudaMallocManaged(&keys_out, size*sizeof(sa_index));


        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            text, keys_out, sa, aux, size);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);


        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            text, keys_out, sa, aux, size);


        cudaDeviceSynchronize();

        //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(sa,aux,n);

        cudaMemcpy(sa, aux, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);

        cudaDeviceSynchronize();
    }

    /*
        \brief Computes two_h_ranks and updates h_ranks according to osipov
        algorithm.
    */
    void update_ranks(size_t h) {
        // Generate 2h-ranks after sorting
        generate_two_h_rank(size, h, sa,
                isa, two_h_rank);

        std::cout << "Tuples with 2h-ranks: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << "<" << sa[i] << "," << h_rank[i] << ","
                << two_h_rank[i] <<">, ";
        }
        std::cout << std::endl;

        //Build Aux
        update_ranks_build_aux(h_rank,
                aux, size);

        std::cout << "Aux after first pass: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;
        //prefix sum over aux
        Max_without_branching max;
        prefix_sum_cub_inclusive(aux, max, size);

        std::cout << "Aux after first pass (prefix sum): ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        //Build aux "tilde"
        update_ranks_build_aux_tilde(
                h_rank, two_h_rank, aux, size);

        std::cout << "Aux after second pass: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        //prefix sum over aux "tilde"
        prefix_sum_cub_inclusive(aux, max, size);

        std::cout << "Aux after second pass(prefix sum): ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;
    }


    /*
        Init ISA with prefix sum method
    */
    void initialize_isa() {

        fill_aux_for_isa(sa, aux, size, cmp);


        Max_without_branching max;

        prefix_sum_cub_inclusive(aux, max, size);


        scatter_to_isa(isa, aux, sa, size);

    }

    /*
        \brief Extracts the suffix index and the h-rank for all considered
        suffixes during this iteration.
    */
    size_t create_tuples(size_t size, size_t h) {
        size_t s=0;
        auto tuple_index = two_h_rank;
        //TODO: Set block_amount and block_size accordingly
        set_tuple(size, h, sa, isa, aux);


        std::cout << "Aux init: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        // Save amount of tuples for last index (gets overwritten by prefix sum)
        s = aux[size-1];
        // Prefix sum
        //exclusive_sum(aux, aux, size);
        // Use h_rank as temporary array as it wasn't needed before

        //TODO: Similar wrapper for ExclusiveSum as for prefix_sum_cub_inclusive
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        // Receive needed temp storage
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, aux, h_rank, size);
        // Allocate needed temp storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, aux, h_rank, size);
        cudaFree(d_temp_storage);

        cudaDeviceSynchronize();

        cudaMemcpy(aux, h_rank, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);

        std::cout << "Aux after prefix sum: ";
        for(size_t i=0; i < size; ++i) {
            std::cout << aux[i] << ", ";
        }
        std::cout << std::endl;

        // Adjust s by amount of tuples from first 'size-1' suffixes.
        s += aux[size-1];
        new_tuple(size, h, sa, isa, aux,
                tuple_index, h_rank);
        cudaDeviceSynchronize();
        /*
            Copy tuple indices from temporary storage in tuple_index/two_h_rank
            to sa.
        */
        cudaMemcpy(sa, tuple_index, size*sizeof(sa_index),
                cudaMemcpyDeviceToDevice);
        return s;
    }

    /*
        \brief Sorts generated tuples via radix sort.
    */
    void stable_sort() {
        //TODO: Again: Wrapper for SortPairs
        auto aux1 = aux;
        // Use two_h_rank as aux2 because it hasn't been filled for this
        // iteration
        auto aux2 = two_h_rank;

         // Determine temporary device storage requirements
         void     *d_temp_storage = NULL;
         size_t   temp_storage_bytes = 0;

         cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
             h_rank, aux1, sa, aux2, size);
         // Allocate temporary storage
         cudaMalloc(&d_temp_storage, temp_storage_bytes);

         // Run sorting operation
         cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
             h_rank, aux1, sa, aux2, size);

         cudaDeviceSynchronize();

         //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(tuple_index,aux1,n);
         cudaMemcpy(sa, aux2, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);
         cudaMemcpy(h_rank, aux1, size*sizeof(sa_index), cudaMemcpyDeviceToDevice);



         //cudaDeviceSynchronize();
         //copy_to_array<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(two_h_ranks,aux2,n);
    }

    void update_container(size_t s) {
        scatter_to_isa(isa, aux, sa, s);
        cudaDeviceSynchronize();

    }

    void finalize(util::span<sa_index> out_sa) {
        isa_to_sa(isa, out_sa.begin(), out_sa.size());
    }


};
