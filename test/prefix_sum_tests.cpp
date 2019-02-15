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
/*
TEST(PrefixSum, mem_eff) {
    std::cout << "Intitializing container." << std::endl;
    container<size_t> container = {1,2,3,4,5,6,7,8};
    //auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());

    
    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
        container[i] = rand();
    }

    auto oper = sum_fct<size_t>();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum_mem_eff<size_t, sum_fct<size_t>>(span<size_t>(container),
            span<size_t>(container2), true, oper, 0);
}*/

TEST(PrefixSum, eff_max) {
    auto container = sacabench::util::container<size_t>{ 1,3,6,4,8,11,2,9,3,10,1,3,2,5,6,7,3,7,32,5 };
    auto result = make_container<size_t>(container.size());
     
    auto add = [&](size_t a, size_t b) {
        return a + b;
    };
    seq_prefix_sum(span<size_t>(container), span<size_t>(result),
            true, add, (size_t)0);
    par_prefix_sum_eff(span<size_t>(container), span<size_t>(container),
            true, add, (size_t)0);
            
    for (size_t i = 0; i < container.size(); ++i) {
        ASSERT_EQ(container[i], result[i]);
    }
}

TEST(PrefixSum, eff_rand_max) {

    srand(time(NULL));

    // Calc prefixsum with threads.
    const size_t len = 36;
    const size_t test_data_len = len*1024*1024;

    std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;

    // Alloc memory.
    //auto container = make_container<size_t>(test_data_len);
    std::cout << "Intitializing container." << std::endl;
    //container<size_t> container = {1,2,3,4,5,6,7,8,9};
    auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());
    auto container3 = make_container<size_t>(container.size());

    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
    	container[i] = rand();
    }
    
    std::copy(container.begin(), container.end(), container2.begin());
    std::copy(container.begin(), container.end(), container3.begin());

    std::cout << "random generation done" << std::endl;


    auto startt = std::chrono::steady_clock::now();

    auto oper = max_fct<size_t>();

    std::cout << "Computing sequentially." << std::endl;
    seq_prefix_sum<size_t, max_fct<size_t>>(span<size_t>(container3),
            span<size_t>(container3), true, oper, 0);

    auto midt = std::chrono::steady_clock::now();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum_eff<size_t, max_fct<size_t>>(span<size_t>(container2),
            span<size_t>(container2), true, oper, 0);

    auto endt = std::chrono::steady_clock::now();
    auto seq_len = std::chrono::duration_cast<std::chrono::milliseconds>(midt - startt).count();
    auto par_len = std::chrono::duration_cast<std::chrono::milliseconds>(endt - midt).count();

    std::cout << "seq: " << seq_len << " ms" << std::endl;
    std::cout << "par: " << par_len << " ms" << std::endl;

    /*std::cout << container2 << std::endl;
    std::cout << container3 << std::endl;*/
    for(size_t i=0; i < container.size(); ++i) {
        // std::cout << "Index: " << i << ", value: " << container2[i] << std::endl;
        ASSERT_EQ(container2[i], container3[i]);
    }
    std::cout << std::endl;
}

TEST(PrefixSum, eff_rand_add) {

    srand(time(NULL));

    // Calc prefixsum with threads.
    const size_t len = 36;
    const size_t test_data_len = len*1024*1024;

    std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;

    // Alloc memory.
    //auto container = make_container<size_t>(test_data_len);
    std::cout << "Intitializing container." << std::endl;
    //container<size_t> container = {1,2,3,4,5,6,7,8,9};
    auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());
    auto container3 = make_container<size_t>(container.size());

    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
    	container[i] = rand();
    }
    
    std::copy(container.begin(), container.end(), container2.begin());
    std::copy(container.begin(), container.end(), container3.begin());

    std::cout << "random generation done" << std::endl;


    auto startt = std::chrono::steady_clock::now();

    auto oper = sum_fct<size_t>();

    std::cout << "Computing sequentially." << std::endl;
    seq_prefix_sum<size_t, sum_fct<size_t>>(span<size_t>(container3),
            span<size_t>(container3), true, oper, 0);

    auto midt = std::chrono::steady_clock::now();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum_eff<size_t, sum_fct<size_t>>(span<size_t>(container2),
            span<size_t>(container2), true, oper, 0);

    auto endt = std::chrono::steady_clock::now();
    auto seq_len = std::chrono::duration_cast<std::chrono::milliseconds>(midt - startt).count();
    auto par_len = std::chrono::duration_cast<std::chrono::milliseconds>(endt - midt).count();

    std::cout << "seq: " << seq_len << " ms" << std::endl;
    std::cout << "par: " << par_len << " ms" << std::endl;

    /*std::cout << container2 << std::endl;
    std::cout << container3 << std::endl;*/
    for(size_t i=0; i < container.size(); ++i) {
        // std::cout << "Index: " << i << ", value: " << container2[i] << std::endl;
        ASSERT_EQ(container2[i], container3[i]);
    }
    std::cout << std::endl;
}

TEST(PrefixSum, correct_sum) {

    srand(time(NULL));

    // Calc prefixsum with threads.
    const size_t len = 36;
    const size_t test_data_len = len*1024*1024;

    std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;

    // Alloc memory.
    //auto container = make_container<size_t>(test_data_len);
    std::cout << "Intitializing container." << std::endl;
    // container<size_t> container = {1,2,3,4,5,6,7,8,9};
    auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());
    auto container3 = make_container<size_t>(container.size());

    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
    	container[i] = rand();
    }

    std::cout << "random generation done" << std::endl;


    auto startt = std::chrono::steady_clock::now();

    auto oper = sum_fct<size_t>();

    std::cout << "Computing sequentially." << std::endl;
    seq_prefix_sum<size_t, sum_fct<size_t>>(span<size_t>(container),
            span<size_t>(container3), true, oper, 0);;

    auto midt = std::chrono::steady_clock::now();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum<size_t, sum_fct<size_t>>(span<size_t>(container),
            span<size_t>(container2), true, oper, 0);

    auto endt = std::chrono::steady_clock::now();
    auto seq_len = std::chrono::duration_cast<std::chrono::milliseconds>(midt - startt).count();
    auto par_len = std::chrono::duration_cast<std::chrono::milliseconds>(endt - midt).count();

    std::cout << "seq: " << seq_len << " ms" << std::endl;
    std::cout << "par: " << par_len << " ms" << std::endl;

    /*std::cout << container2 << std::endl;
    std::cout << container3 << std::endl; */
    for(size_t i=0; i < container.size(); ++i) {
        // std::cout << "Index: " << i << ", value: " << container2[i] << std::endl;
        ASSERT_EQ(container2[i], container3[i]);
    }
    std::cout << std::endl;
}

TEST(PrefixSum, correct_max) {

    srand(time(NULL));

    // Calc prefixsum with threads.
    const size_t len = 36;
    const size_t test_data_len = len*1024*1024;

    std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;

    // Alloc memory.
    //auto container = make_container<size_t>(test_data_len);
    std::cout << "Intitializing container." << std::endl;
    //container<size_t> container = {1,2,3,4,5,6,7,8,9};
    auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());
    auto container3 = make_container<size_t>(container.size());

    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
    	container[i] = rand();
    }

    std::cout << "random generation done" << std::endl;


    auto startt = std::chrono::steady_clock::now();

    auto oper = max_fct<size_t>();

    std::cout << "Computing sequentially." << std::endl;
    seq_prefix_sum<size_t, max_fct<size_t>>(span<size_t>(container),
            span<size_t>(container3), true, oper, 0);;

    auto midt = std::chrono::steady_clock::now();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum<size_t, max_fct<size_t>>(span<size_t>(container),
            span<size_t>(container2), true, oper, 0);

    auto endt = std::chrono::steady_clock::now();
    auto seq_len = std::chrono::duration_cast<std::chrono::milliseconds>(midt - startt).count();
    auto par_len = std::chrono::duration_cast<std::chrono::milliseconds>(endt - midt).count();

    std::cout << "seq: " << seq_len << " ms" << std::endl;
    std::cout << "par: " << par_len << " ms" << std::endl;

    /*std::cout << container2 << std::endl;
    std::cout << container3 << std::endl;*/
    for(size_t i=0; i < container.size(); ++i) {
        // std::cout << "Index: " << i << ", value: " << container2[i] << std::endl;
        ASSERT_EQ(container2[i], container3[i]);
    }
    std::cout << std::endl;
}

TEST(PrefixSum, eff_rand_max_non_inclusive) {

    srand(time(NULL));

    // Calc prefixsum with threads.
    const size_t len = 36;
    const size_t test_data_len = len*1024*1024;

    std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;

    // Alloc memory.
    //auto container = make_container<size_t>(test_data_len);
    std::cout << "Intitializing container." << std::endl;
    //container<size_t> container = {1,2,3,4,5,6,7,8,9};
    auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());
    auto container3 = make_container<size_t>(container.size());

    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
    	container[i] = rand();
    }
    
    std::copy(container.begin(), container.end(), container2.begin());
    std::copy(container.begin(), container.end(), container3.begin());

    std::cout << "random generation done" << std::endl;


    auto startt = std::chrono::steady_clock::now();

    auto oper = max_fct<size_t>();

    std::cout << "Computing sequentially." << std::endl;
    seq_prefix_sum<size_t, max_fct<size_t>>(span<size_t>(container3),
            span<size_t>(container3), false, oper, 0);;

    auto midt = std::chrono::steady_clock::now();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum_eff<size_t, max_fct<size_t>>(span<size_t>(container2),
            span<size_t>(container2), false, oper, 0);

    auto endt = std::chrono::steady_clock::now();
    auto seq_len = std::chrono::duration_cast<std::chrono::milliseconds>(midt - startt).count();
    auto par_len = std::chrono::duration_cast<std::chrono::milliseconds>(endt - midt).count();

    std::cout << "seq: " << seq_len << " ms" << std::endl;
    std::cout << "par: " << par_len << " ms" << std::endl;

    /*std::cout << container2 << std::endl;
    std::cout << container3 << std::endl;*/
    for(size_t i=0; i < container.size(); ++i) {
        // std::cout << "Index: " << i << ", value: " << container2[i] << std::endl;
        ASSERT_EQ(container2[i], container3[i]);
    }
    std::cout << std::endl;
}

TEST(PrefixSum, eff_rand_add_non_inclusive) {

    srand(time(NULL));

    // Calc prefixsum with threads.
    const size_t len = 36;
    const size_t test_data_len = len*1024*1024;

    std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;

    // Alloc memory.
    //auto container = make_container<size_t>(test_data_len);
    std::cout << "Intitializing container." << std::endl;
    //container<size_t> container = {1,2,3,4,5,6,7,8,9};
    auto container = make_container<size_t>(test_data_len);
    auto container2 = make_container<size_t>(container.size());
    auto container3 = make_container<size_t>(container.size());

    std::cout << "Generating random numbers." << std::endl;
    for(size_t i = 0; i < test_data_len; ++i) {
    	container[i] = rand();
    }
    
    std::copy(container.begin(), container.end(), container2.begin());
    std::copy(container.begin(), container.end(), container3.begin());

    std::cout << "random generation done" << std::endl;


    auto startt = std::chrono::steady_clock::now();

    auto oper = sum_fct<size_t>();

    std::cout << "Computing sequentially." << std::endl;
    seq_prefix_sum<size_t, sum_fct<size_t>>(span<size_t>(container3),
            span<size_t>(container3), false, oper, 0);;

    auto midt = std::chrono::steady_clock::now();

    std::cout << "Computing parallelly." << std::endl;
    par_prefix_sum_eff<size_t, sum_fct<size_t>>(span<size_t>(container2),
            span<size_t>(container2), false, oper, 0);

    auto endt = std::chrono::steady_clock::now();
    auto seq_len = std::chrono::duration_cast<std::chrono::milliseconds>(midt - startt).count();
    auto par_len = std::chrono::duration_cast<std::chrono::milliseconds>(endt - midt).count();

    std::cout << "seq: " << seq_len << " ms" << std::endl;
    std::cout << "par: " << par_len << " ms" << std::endl;

    /*std::cout << container2 << std::endl;
    std::cout << container3 << std::endl;*/
    for(size_t i=0; i < container.size(); ++i) {
        // std::cout << "Index: " << i << ", value: " << container2[i] << std::endl;
        ASSERT_EQ(container2[i], container3[i]);
    }
    std::cout << std::endl;
}
