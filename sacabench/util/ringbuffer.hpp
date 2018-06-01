/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/span.hpp>

namespace sacabench::util {
template <typename content>
class ringbuffer {
public:
    inline ringbuffer(const size_t capacity)
        : memory(make_container<content>(capacity)), start(0),
          end(capacity - 1) {}

    inline void push_front(const content e) {
        DCHECK_EQ(is_full(), false);
        memory[end] = e;
        end--;
    }

    inline void push_back(const content e) {
        DCHECK_EQ(is_full(), false);
        memory[start] = e;
        start++;
    }

    inline void traverse(const util::span<content> out) const {
        const auto wrapping_add_one = [&](size_t a) {
            return (a + 1) % size();
        };

        for (size_t idx = wrapping_add_one(end), i = 0; i < size();
             idx = wrapping_add_one(idx), ++i) {
            out[i] = memory[idx];
        }
    }

    inline size_t size() const { return memory.size(); }

    inline bool is_full() const { return start == (end + 1) % size(); }

private:
    util::container<content> memory;
    size_t start;
    size_t end;
};
} // namespace sacabench::util
