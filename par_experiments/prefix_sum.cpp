#include <iostream>

struct Node {
    size_t from_left;
    size_t sum;
}

// Needed for down pass
size_t left_child(size_t parent) { return 2*parent + 1; }

size_t right_child(size_t parent) { return 2*parent+2; }

// Needed for up pass
size_t parent_from_left_child(size_t child) { return (parent-1)/2; }

size_t parent_from_right_child(size_t child) { return (parent-2)/2; }

template<typename Content>
void par_prefix_sum(const Content* in, const size_t len, Content* out) {
    // Correct length for odd lengths.
    size_t corrected_len = (len % 2 == 1) ? len + 1 : len;
    
    // Create tree leafs
    auto tree = new Content[2 * corrected_len];
    memcpy(tree + corrected_len, in, len);
    
    //#pragma omp parallel for
    for(size_t i = 0; i < corrected_len; i += 2) {
        
    }
}

int main() {
    
}
