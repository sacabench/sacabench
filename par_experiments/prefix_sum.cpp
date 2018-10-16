#include <iostream>

struct node {
    size_t from_left;
    size_t sum;
}

// Needed for down pass
size_t left_child(int parent) { return 2*parent + 1; }

size_t right_child(int parent) { return 2*parent+2; }


// Needed for up pass
size_t parent_from_left_child(int child) { return (parent-1)/2; }

size_t parent_from_right_child(int child) { return (parent-2)/2; }
