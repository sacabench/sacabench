// Komilieren: nvcc Prefix_Sum_Benchmark.cu -std=c++11 -o output
// Kompilieren mit Optimierung:  nvcc -Xptxas -O3,-v Prefix_Sum_Benchmark.cu -std=c++11 -O3 -D_FORCE_INLINES -o output

#include "cub-1.8.0/cub/cub.cuh"
#include <iostream>
#include <stdlib.h> 

//Sacabench includes
#include <iostream>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include <chrono>
#include <memory>
#include <ctime>
#include <cstdint>




// Operator for sum-operation (function call for prefix-sum)
template <typename Content>
struct sum_fct{
public:
    // elem and compare_to need to be smaller than input.size()
    inline Content operator()(const Content& in1, const Content& in2) const {
        return in1 + in2;
    }
};

// Operator for max-operation (function call for prefix-sum)
template <typename Content>
struct max_fct{
public:
    // elem and compare_to need to be smaller than input.size()
    inline Content operator()(const Content& in1, const Content& in2) const {
        return std::max(in1, in2);
    }
};


// Sequential version of the prefix sum calculation.
template <typename Content, typename add_operator>
void seq_prefix_sum(int* in, int* out, bool inclusive,
        add_operator add, Content identity, uint64_t N) {
    if(inclusive) {
        out[0] = add(identity, in[0]);
        for(size_t i = 1; i < N; ++i) {
            out[i] = add(in[i], out[i - 1]);
        }
    } else {
        Content tmp2, tmp = in[0];
        out[0] = identity;
        for(size_t i=1; i < N; ++i) {
            tmp2 = in[i];
            out[i] = add(tmp, out[i-1]);
            tmp = tmp2;
        }
    }
    std::cout<<"Last Element (seq): "<<out[N-1]<<std::endl;

}

inline size_t next_power_of_two(size_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;

	return v;
}
inline size_t prev_power_of_two(size_t v) {
    v = next_power_of_two(v);
    return v >> 1;
}

// Needed for down pass
inline size_t left_child(size_t parent) { return 2*parent; }
inline size_t right_child(size_t parent) { return 2*parent+1; }
inline size_t parent_of_child(size_t child) { return child/2; }


template <typename Content, typename add_operator>
void par_prefix_sum(int* in, int* out, bool inclusive,
        add_operator add, Content identity, uint64_t N) {
    const size_t corrected_len = (N % 2 == 1) ? N + 1 : N;
    const size_t tree_size = next_power_of_two(N);
    int* tree = new int[corrected_len + tree_size];
    //std::cout << "tree size: " << tree.size() << std::endl;

    // copy values of in into last level of tree
    //#pragma omp parallel for schedule(dynamic, 2048)
    #pragma omp simd
    for(size_t i=0; i < N; ++i) {
        tree[tree_size + i] = in[i];
    }

    // Up-Pass

    for(size_t offset = tree_size; offset != 1; offset /= 2) {
        #pragma omp parallel for schedule(dynamic, 2048) shared(tree)
        for(size_t i = 0; i < std::min(offset, corrected_len); i += 2) {
            const size_t j = offset + i;
            const size_t k = offset + i + 1;
            tree[parent_of_child(j)] = add(tree[j], tree[k]);
        }
    }

    // First element needs to be set to identity
    tree[1] = identity;
    // Downpass
    for(size_t offset = 1; offset != tree_size; offset *= 2) {

        const size_t layer_size = offset == tree_size / 2 ? corrected_len / 2 : offset;

        #pragma omp parallel for schedule(dynamic, 2048) shared(tree)
        for(size_t i = 0; i < layer_size; i++) {
            const size_t j = layer_size - i - 1;

            const Content& from_left = tree[offset + j];
            const Content& left_sum = tree[left_child(offset + j)];
            tree[right_child(offset + j)] = add(from_left, left_sum);
            tree[left_child(offset + j)] = from_left;
        }
    }

    // ########################################################
    // FINAL PASS (Copy to out)
    // ########################################################

    // Prefer branch outside of simd-loop
    if(inclusive) {
        #pragma omp simd
        for(size_t i = 0; i < N; ++i) {
            out[i] = add(in[i], tree[tree_size + i]);
        }
    } else {
        #pragma omp simd
        for(size_t i = 0; i < N; ++i) {
            out[i] = tree[tree_size + i];
        }
    }
    std::cout<<"Last Element (par): "<<out[N-1]<<std::endl;

}



void prefix_sum_cub_inclusive(int* values_in, int* values_out, uint64_t N)
{
    
    
        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, values_in, values_out, N);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run inclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, values_in, values_out, N);

        cudaDeviceSynchronize();


        std::cout<<"Last Element (gpu): "<<values_out[N-1]<<std::endl;
    }


int my_main(uint64_t n)
{
    std::cout<<"Start initialization"<<std::endl;
//    int n = 1000*1000;
    
    std::cout << "size: " << n << std::endl;
    
    //Init arrays
    int* cpu_array_in = new int[n];
    int* cpu_array_out = new int[n];
    int* gpu_array_in;
    int* gpu_array_out;



    //Allocate Arrays in GPU Memory
    cudaMallocManaged(&gpu_array_in, n*sizeof(int));
    cudaMallocManaged(&gpu_array_out, n*sizeof(int));
    std::cout<<"Fill arrays"<<std::endl;
    srand(time(NULL));
    int random;
    //befülle arrays mit zufallszahlen
    for(uint64_t index = 0; index < n;++index)
    {
        random = rand() % 1000000;
        cpu_array_in[index]= random;
        gpu_array_in[index]= random;
    }
    std::cout<<"Start calculating"<<std::endl;

    //Use CUDA API for time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //Run CUDA Version
    prefix_sum_cub_inclusive(gpu_array_in,gpu_array_out, n);  
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout<<"Elapsed Time on GPU: "<<milliseconds<<" ms"<<std::endl;


    cudaEventRecord(start);
    //Run Seq CPU Version
    seq_prefix_sum<int, sum_fct<int>>(cpu_array_in, cpu_array_out, true, sum_fct<int>(), 0,n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout<<"Elapsed Time on CPU: "<<milliseconds<<" ms"<<std::endl;

    cudaEventRecord(start);
    //Run Par CPU Version
    par_prefix_sum<int, sum_fct<int>>(cpu_array_in, cpu_array_out, true, sum_fct<int>(), 0,n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout<<"Elapsed Time on CPU (par): "<<milliseconds<<" ms"<<std::endl;


    cudaFree(gpu_array_in);
    cudaFree(gpu_array_out);
    return 0;

}

int main() {
  uint64_t v = 1000ull;
  for (int i = 0; i < 4; i++) {
    my_main(v);
    v *= 1000ull;
    std::cout << std::endl;
  }
}
