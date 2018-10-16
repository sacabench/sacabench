#include <iostream>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include <chrono>
#include <memory>

template <typename Fn>
double duration(Fn fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const auto dur = end - start;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
}

// Sequential version of the prefix sum calculation.
template<typename Content>
void seq_prefixsum(const Content* __restrict__ in, const size_t len, Content* __restrict__ out) {
    out[0] = in[0];
    for(size_t i = 1; i < len; ++i) {
        out[i] = in[i] + out[i - 1];
    }
}

size_t next_power_of_two(size_t v) {	
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

// Needed for down pass
size_t left_child(size_t parent) { return 2*parent; }

size_t right_child(size_t parent) { return 2*parent+1; }

// Needed for up pass
size_t parent_of_child(size_t child) { return child/2; }

// Parallel version of the prefix sum calculation.
template<typename Content>
void par_prefix_sum(const Content* in, const size_t len, Content* out) {
    // Correct length for odd lengths.
    const size_t corrected_len = (len % 2 == 1) ? len + 1 : len;
    const size_t tree_size = next_power_of_two(len);
    
    // Create tree structure
    auto tree = new Content[tree_size + corrected_len];
    tree[0] = 42; // lol
    
    // Copy input into tree leafs
    memcpy(tree + tree_size, in, len * sizeof(Content));

    // ########################################################    
    // UP PASS
    // ########################################################
    
    for(size_t offset = tree_size; offset != 1; offset /= 2) {
    
        #pragma omp parallel for schedule(dynamic, 128)
        for(size_t i = 0; i < std::min(offset, corrected_len); i += 2) {
            const size_t j = offset + i;
            const size_t k = offset + i + 1;
            tree[parent_of_child(j)] = tree[j] + tree[k];
        }
    }
    
    std::cout << "up pass done" << std::endl;
    
    /*for(size_t i = 0; i < tree_size + corrected_len; ++i) {
        std::cout << tree[i] << " ";
    }
    std::cout << std::endl;*/
    
    // ########################################################    
    // DOWN PASS
    // ########################################################
    
    tree[1] = 0;
    
    for(size_t offset = 1; offset != tree_size; offset *= 2) { 
    
        const size_t layer_size = offset == tree_size / 2 ? corrected_len / 2 : offset;
    
        #pragma omp parallel for schedule(dynamic, 128)
        for(size_t i = 0; i < layer_size; i++) {
            const size_t j = layer_size - i - 1;
            
            const Content& from_left = tree[offset + j];
            const Content& left_sum = tree[left_child(offset + j)];
            tree[right_child(offset + j)] = from_left + left_sum;
            tree[left_child(offset + j)] = from_left;
        }
    }
    
/*    for(size_t i = 0; i < tree_size + corrected_len; ++i) {
        std::cout << tree[i] << " ";
    }
    std::cout << std::endl;*/
    
    // ########################################################    
    // FINAL PASS
    // ########################################################
    
    #pragma omp parallel for
    for(size_t i = 0; i < len; ++i) {
        out[i] = in[i] + tree[tree_size + i];
    }
}

int main() {

  srand(time(NULL));

  // Number of CPUs.
  //const size_t n_cpus = std::thread::hardware_concurrency();
  //omp_set_num_threads(n_cpus);

  // Calc prefixsum with threads.
  const size_t len = 90;
  const size_t test_data_len = len*1024*1024;
  
  std::cout << "memory: " << (test_data_len * sizeof(size_t) / 1024.0 / 1024.0) << "MiB" << std::endl;
  
  // Alloc memory.
  auto container = std::make_unique<size_t[]>(test_data_len);
  auto container2 = std::make_unique<size_t[]>(test_data_len);
  auto container3 = std::make_unique<size_t[]>(test_data_len);
  auto ptr = container.get();
  auto ptr2 = container2.get();
  auto ptr3 = container3.get();
  
  #pragma omp parallel for schedule(static, 256)
  for(size_t i = 0; i < test_data_len; ++i) {
  	ptr[i] = rand();
  }
   
  std::cout << "random generation done" << std::endl;
   
  //print_arr(ptr, test_data_len);

  auto startt = std::chrono::steady_clock::now();

  seq_prefixsum(ptr, test_data_len, ptr3);
  
  auto midt = std::chrono::steady_clock::now();
  
  par_prefix_sum(ptr, test_data_len, ptr2);

  auto endt = std::chrono::steady_clock::now();
  
  std::cout << "seq: " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(midt - startt).count()) << std::endl;
  std::cout << "par: " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(endt - midt).count()) << std::endl;

  //print_arr(ptr2, test_data_len);
  //print_arr(ptr3, test_data_len);
    
  for(size_t i = 0; i < test_data_len; ++i) {
    if(ptr2[i] != ptr3[i]) {
      std::cout << "fehler: " << ptr2[i] << "!=" << ptr3[i] << std::endl;
      exit(-1);
    }
  }
  
  // Done
  return 0;
}
