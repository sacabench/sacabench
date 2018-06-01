/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <random>
#include <util/container.hpp>
#include <util/ringbuffer.hpp>

using namespace sacabench::util;

TEST(ringbuffer, create) {
    ringbuffer<size_t> rb(16);
    ASSERT_EQ(rb.size(), 16);
}

TEST(ringbuffer, push) {
    ringbuffer<size_t> rb(16);

    for (size_t i = 5; i < 16; ++i) {
        rb.push_back(i);
    }

    for (size_t i = 0; i < 5; ++i) {
        size_t j = 4 - i;
        rb.push_front(j);
    }

    auto c = make_container<size_t>(16);
    rb.traverse(c);

    for (size_t i = 0; i < 16; ++i) {
        ASSERT_EQ(c[i], i);
    }
}

TEST(ringbuffer, push2) {
    ringbuffer<size_t> rb(16);

    for (size_t i = 5; i < 10; ++i) {
        rb.push_back(i);
    }

    for (size_t i = 0; i < 5; ++i) {
        size_t j = 4 - i;
        rb.push_front(j);
    }

    auto c = make_container<size_t>(16);
    rb.traverse(c);

    for (size_t i = 0; i < 10; ++i) {
        ASSERT_EQ(c[i], i);
    }
}

TEST(ringbuffer, full) {
    ringbuffer<size_t> rb(16);

    for (size_t i = 0; i < 16; ++i) {
        rb.push_back(i);
    }

    bool back_exception = false, front_exception = false;

    try {
        rb.push_back(40);
    } catch (...) {
        back_exception = true;
        return;
    }

    try {
        rb.push_front(40);
    } catch (...) {
        front_exception = true;
        return;
    }

    ASSERT_TRUE(back_exception);
    ASSERT_TRUE(front_exception);
}

TEST(ringbuffer, random) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(false, true);

    for (size_t i = 0; i < 100000; ++i) {
        ringbuffer<size_t> rb(5);
        for (size_t j = 0; j < 5; ++j) {
            bool use_front = dist(gen);
            if (use_front) {
                rb.push_front(i);
            } else {
                rb.push_back(i);
            }
        }
        ASSERT_EQ(rb.size(), 5);
    }
}
