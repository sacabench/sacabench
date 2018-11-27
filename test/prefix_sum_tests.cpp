#include <gtest/gtest.h>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include <chrono>
#include <memory>
#include <util/span.hpp>
#include <util/container.hpp>
#include <util/prefix_sum.hpp>

using namespace sacabench::util;

TEST(PrefixSum, correct_sum) {

    srand(time(NULL));

    // Number of CPUs.
    //const size_t n_cpus = std::thread::hardware_concurrency();
    //omp_set_num_threads(n_cpus);

    // Calc prefixsum with threads.
    const size_t len = 36;
    const size_t test_data_len = len*1024*1024;

    std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;

    // Alloc memory.
    //auto container = make_container<size_t>(test_data_len);
    std::cout << "Intitializing container." << std::endl;
    //container<size_t> container {1,2,3,4,5,6,7,8,9};
    auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());
    auto container3 = make_container<size_t>(container.size());
    /*auto ptr = container.begin();
    auto ptr2 = container2.begin();
    auto ptr3 = container3.begin();
    */

    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
    	container[i] = rand();
    }


    std::cout << "random generation done" << std::endl;

    //print_arr(ptr, test_data_len);

    auto startt = std::chrono::steady_clock::now();

    std::cout << "Computing sequentially." << std::endl;
    seq_prefix_sum<size_t, sum<size_t>>(span<size_t>(container),
            span<size_t>(container3), true, sum<size_t>(), 0);;

    auto midt = std::chrono::steady_clock::now();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum<size_t, sum<size_t>>(span<size_t>(container),
            span<size_t>(container2), true, sum<size_t>(), 0);

    auto endt = std::chrono::steady_clock::now();

    std::cout << "seq: " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(midt - startt).count()) << std::endl;
    std::cout << "par: " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(endt - midt).count()) << std::endl;

    //print_arr(ptr2, test_data_len);
    //print_arr(ptr3, test_data_len);
    /*std::cout << container2 << std::endl;
    std::cout << container3 << std::endl;*/
    for(size_t i=0; i < container.size(); ++i) {
        // std::cout << "Index: " << i << ", value: " << container2[i] << std::endl;
        ASSERT_EQ(container2[i], container3[i]);
    }
    std::cout << std::endl;
}
