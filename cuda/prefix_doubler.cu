#include <string>
#include <iostream>
#include "cub-1.8.0/cub/cub.cuh"



    __global__
    static void initialize_sa_gpu(int n, int*  sa) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        
        for (int i = index; i < n; i+=stride) {
            sa[i] = i;
        }
            
    }

    static void inital_sorting(const char* text, int* sa, int n) {
     //Tatsächliche Werte
    char  *keys_out;     // e.g., [        ...        ]
    //Indizes
    int  *values_out;   // e.g., [        ...        ]



    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&keys_out, n*sizeof(char));
    cudaMallocManaged(&values_out, n*sizeof(int));


    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        text, keys_out, sa, values_out, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);


    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        text, keys_out, sa, values_out, n);



    }


    static void prefix_doubling_gpu(const char* text, int* out_sa, int n) {
        
        int* sa;
        //Wofür??
        int* isa_container;
        int* aux_container;
        cudaMallocManaged(&sa, n*sizeof(int));
        cudaMallocManaged(&isa_container, n*sizeof(int));
        cudaMallocManaged(&aux_container, n*sizeof(int));

        initialize_sa_gpu<<<2,4>>>(n, sa);

        cudaDeviceSynchronize();

        inital_sorting(text, sa, n);

        cudaDeviceSynchronize();


        /*
        std::cout<<"Hallo"<<std::endl;
        for(int index = 0; index < n; ++index) {
            std::cout<<sa[index]<<", ";
        }
        std::cout<<std::endl;
        */

        //int h = 4;
        // Sort by h characters
        //compare_first_four_chars cmp_init = compare_first_four_chars(text);

        
        //Initiale Sortierung
        //Möglichkeit 1: mit Thrust sortieren > direkt mit Key Funktion nutzbar aber langsamer als CUB -> Thrust Vectoren benötigt, meh
        //Möglichkeit 2: CUB nach nur einem Buchstaben -> Meh
        //Möglichkeit 3: CUB, aber vorher den Text mittels wordpacking transformieren
        //util::sort::ips4o_sort_parallel(sa, cmp_init);
/*      initialize_isa<sa_index, compare_first_four_chars>(sa, isa, aux,
                                                           cmp_init);
        phase.split("Mark singletons");
        mark_singletons(sa, isa);
        phase.split("Loop Initialization");

        // std::cout << "isa: " << isa << std::endl;
        size_t size = sa.size();
        size_t s = 0;

        auto tuple_container =
            util::make_container<std::tuple<sa_index, sa_index, sa_index>>(
                size);
        util::span<std::tuple<sa_index, sa_index, sa_index>> tuples;
        compare_tuples<sa_index> cmp;
        while (size > 0) {
            phase.split("Iteration");
            aux = util::span<sa_index>(aux_container).slice(0, size);
            tuples = tuple_container.slice(0, size);

            //s = create_tuples<sa_index>(tuples.slice(0, size), size, h, sa, isa);
            s = create_tuples_parallel<sa_index>(tuples.slice(0, size),
                    size, h, sa, isa, aux);
            //std::cout << "Elements left: " << size << std::endl;

            // std::cout << "Next size: " << s << std::endl;
            // Skip all operations till size gets its new size, if this
            // iteration contains no tuples
            if (s > 0) {
                tuples = tuples.slice(0, s);
                aux = util::span<sa_index>(aux).slice(0, s);
                // std::cout << "Sorting tuples." << std::endl;
                cmp = compare_tuples<sa_index>(tuples);
                util::sort::std_par_stable_sort(tuples, cmp);
                sa = sa.slice(0, s);
                update_ranks_prefixsum(tuples, aux);
                // std::cout << "Writing new order to sa." << std::endl;
                for (size_t i = 0; i < s; ++i) {
                    sa[i] = std::get<0>(tuples[i]);
                }

                for (size_t i = 0; i < s; ++i) {

                    isa[std::get<0>(tuples[i])] =
                        aux[i]; // std::get<1>(tuples[i]);
                }
                mark_singletons(sa, isa);
            }
            size = s;
            h = 2 * h;
        }
        phase.split("Write out SA");
        for (size_t i = 0; i < out_sa.size(); ++i) {
            out_sa[isa[i] ^ utils<sa_index>::NEGATIVE_MASK] = i;
        }
        */
    }
int main()
{
    std::string text_str = "caabaccaabacaa";
    const char* text = text_str.c_str();
    int n = text_str.size();
    std::cout<<"n: "<<n<<std::endl;

    int* out_sa;
    cudaMallocManaged(&out_sa, n*sizeof(int));

    prefix_doubling_gpu(text, out_sa, n);
    return 0;
}
