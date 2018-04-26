#pragma once

#include <inttypes>

namespace util {
namespace sort {
    // Sort the suffix indices in array by comparing one character in
    // input_text.
    void multikey_quicksort(span<index_type> array, const input_type& input_text);

    // Create a function with compares at one character depth.
    struct compare_one_character_at_depth {
    public:
        // The depth at which we compare.
        index_type depth;

        // 0 if equal, -1 if the first is smaller, 1 if the first is larger.
        int compare(const index_type&, const index_type&) const noexcept;
    private:
        // A reference to the input text.
        input_type& input_text;
    }
}
}
