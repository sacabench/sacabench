/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <random>
#include <util/container.hpp>
#include <util/ringbuffer.hpp>
#include <util/macros.hpp>
#include <util/is_sorted.hpp>

using namespace sacabench::util;

TEST(ringbuffer, create) {
    auto memory = make_container<size_t>(16);
    ringbuffer<size_t> rb(memory);
    ASSERT_EQ(rb.size(), 0u);
    ASSERT_EQ(rb.capacity(), 16u);
}

TEST(ringbuffer, push) {
    auto memory = make_container<size_t>(16);
    ringbuffer<size_t> rb(memory);

    ASSERT_EQ(rb.size(), 0u);
    ASSERT_EQ(rb.capacity(), 16u);

    rb.push_back(5);
    rb.push_back(6);

    rb.push_front(4);
    rb.push_front(3);
    rb.push_front(2);
    rb.push_front(1);

    rb.push_back(7);
}

TEST(ringbuffer, push_and_foreach) {
    auto memory = make_container<size_t>(16);
    ringbuffer<size_t> rb(memory);

    ASSERT_EQ(rb.size(), 0u);
    ASSERT_EQ(rb.capacity(), 16u);

    rb.push_back(5);
    rb.push_back(6);

    rb.push_front(4);
    rb.push_front(3);
    rb.push_front(2);
    rb.push_front(1);

    rb.push_back(7);

    auto sorted_memory = std::vector<size_t>();
    rb.for_each([&](auto e) {
        sorted_memory.push_back(e);
    });

    ASSERT_TRUE(is_sorted(span(sorted_memory)));
}

TEST(ringbuffer, push_and_copy_into) {
    auto memory = make_container<size_t>(16);
    ringbuffer<size_t> rb(memory);

    ASSERT_EQ(rb.size(), 0u);
    ASSERT_EQ(rb.capacity(), 16u);

    rb.push_back(5);
    rb.push_back(6);

    rb.push_front(4);
    rb.push_front(3);
    rb.push_front(2);
    rb.push_front(1);

    rb.push_back(7);

    auto sorted_memory = make_container<size_t>(7);
    rb.copy_into(sorted_memory);

    ASSERT_TRUE(is_sorted(span<size_t>(sorted_memory)));
}

TEST(ringbuffer, random) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(false, true);

    auto memory = make_container<size_t>(5);

    for (size_t i = 0; i < 1000000; ++i) {
        ringbuffer<size_t> rb(memory);
        for (size_t j = 0; j < 5; ++j) {
            bool use_front = dist(gen);
            if (use_front) {
                rb.push_front(j);
            } else {
                rb.push_back(j);
            }
        }

        ASSERT_EQ(rb.size(), 5u);
    }
}
