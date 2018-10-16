#include <iostream>
#include <cstring>

struct Node {
    size_t from_left;
    size_t sum;
};

// Needed for down pass
size_t left_child(size_t parent) { return 2*parent; }

size_t right_child(size_t parent) { return 2*parent+1; }

// Needed for up pass
size_t parent_of_child(size_t child) { return child/2; }

template<typename Content>
void par_prefix_sum(const Content* in, const size_t len, Content* out) {
    // Correct length for odd lengths.
    size_t corrected_len = (len % 2 == 1) ? len + 1 : len;
    
    // Create tree leafs
    auto tree = new Content[2 * corrected_len];
    tree[0] = 0x42; // lol
    memcpy(tree + corrected_len, in, len);
    
    //#pragma omp parallel for
    for(size_t i = 0; i < corrected_len; i += 2) {
        const size_t j = corrected_len + i;
        const size_t k = corrected_len + i + 1;
        tree[parent_of_child(j)] = tree[j] + tree[k];
    }
    
    std::cout << "up pass done" << std::endl;
    
    for(size_t i = 0; i < 2*corrected_len; ++i) {
        std::cout << tree[i];
    }
    std::cout << std::endl;
}

int main() {
    const size_t test_data[] = { 6,4,16,10,16,14,2,8 };
    size_t out[8];
    par_prefix_sum(test_data, 8, out);
}
