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
    inline ringbuffer(util::span<content> _memory) : memory(_memory), front(), back() {}

    inline void push_back(const content e) {
        DCHECK(!is_full());
        back = memory.slice(0, back.size() + 1);
        back[back.size() - 1] = std::move(e);
    }

    inline void push_front(const content e) {
        DCHECK(!is_full());
        front = memory.slice(capacity() - (front.size() + 1));
        front[0] = std::move(e);
    }

    inline bool is_full() const {
        return size() >= capacity();
    }

    inline size_t capacity() const {
        return memory.size();
    }

    inline size_t size() const {
        return front.size() + back.size();
    }

    inline void print() const {
        std::cout << memory << std::endl;
        std::cout << front << back << std::endl;
    }

    template<typename Fn>
    inline void for_each(const Fn fn) const {
        for(const content& e : front) {
            fn(e);
        }
        for(const content& e : back) {
            fn(e);
        }
    }

    inline void copy_into(util::span<content> s) const {
        size_t i = 0;
        DCHECK_LE(size(), s.size());
        for_each([&](const content& e){
            s[i] = e;
            ++i;
        });
    }

private:
    util::span<content> memory;
    util::span<content> front;
    util::span<content> back;
};
} // namespace sacabench::util
