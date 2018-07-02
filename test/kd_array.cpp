/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/kd_array.hpp>

using namespace sacabench::util;

TEST(kd_array, create) {
    { kd_array<size_t, 2> array({10, 10}); }
    { kd_array<size_t, 3> array({10, 10, 10}); }
    { kd_array<size_t, 4> array({10, 10, 10, 10}); }
    { kd_array<size_t, 5> array({10, 10, 10, 10, 10}); }
}

TEST(kd_array, index2d) {
    kd_array<size_t, 2> array({10, 10});

    size_t i = 0;

    for (size_t a = 0; a < 10; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            ASSERT_EQ(i, array.index({a, b}));
            ++i;
        }
    }
}

TEST(kd_array, direct_access_set) {
    kd_array<size_t, 3> array({10, 10, 10});

    size_t i = 0;

    for (size_t a = 0; a < 10; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            for (size_t c = 0; c < 10; ++c) {
                array.get_mut_direct_unsafe(i) = i;
                ASSERT_EQ(i, array[a][b][c]);
                ++i;
            }
        }
    }
}

TEST(kd_array, direct_access_get) {
    kd_array<size_t, 3> array({10, 10, 10});

    size_t i = 0;

    for (size_t a = 0; a < 10; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            for (size_t c = 0; c < 10; ++c) {
                array[a][b][c] = i;
                ASSERT_EQ(i, array.get_direct_unsafe(i));
                ++i;
            }
        }
    }
}

TEST(kd_array, index3d) {
    kd_array<size_t, 3> array({10, 10, 10});

    size_t i = 0;

    for (size_t a = 0; a < 10; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            for (size_t c = 0; c < 10; ++c) {
                ASSERT_EQ(i, array.index({a, b, c}));
                ++i;
            }
        }
    }
}

TEST(kd_array, index3d_different_sizes) {
    kd_array<size_t, 3> array({5, 10, 15});

    size_t i = 0;
    for (size_t a = 0; a < 5; ++a) {
        for (size_t b = 0; b < 10; ++b) {
            for (size_t c = 0; c < 15; ++c) {
                ASSERT_EQ(i, array.index({a, b, c}));
                ++i;
            }
        }
    }
}

TEST(kd_array, access3d) {
    kd_array<size_t, 3> array({5, 10, 15});

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    array.set({a, b, c}, i);
                    ++i;
                }
            }
        }
    }

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    ASSERT_EQ(array.get({a, b, c}), i);
                    ++i;
                }
            }
        }
    }
}

TEST(kd_array, access3d_better_syntax) {
    kd_array<size_t, 3> array({5, 10, 15});

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    array.set({a, b, c}, i);
                    ++i;
                }
            }
        }
    }

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    ASSERT_EQ(i, array[a][b][c]);
                    ++i;
                }
            }
        }
    }
}

TEST(kd_array, access3d_better_syntax_const) {
    kd_array<size_t, 3> array({5, 10, 15});

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    array.set({a, b, c}, i);
                    ++i;
                }
            }
        }
    }

    const kd_array<size_t, 3>& array2 =
        const_cast<const kd_array<size_t, 3>&>(array);

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    ASSERT_EQ(i, array2[a][b][c]);
                    ++i;
                }
            }
        }
    }
}

TEST(kd_array, access3d_set_better_syntax) {
    kd_array<size_t, 3> array({5, 10, 15});

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    array[a][b][c] = i;
                    ++i;
                }
            }
        }
    }

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    ASSERT_EQ(i, array[a][b][c]);
                    ++i;
                }
            }
        }
    }
}

// TEST(kd_array, access3d_set_on_const) {
//     const kd_array<size_t, 3> array({5, 10, 15});
//
//     {
//         size_t i = 0;
//         for (size_t a = 0; a < 5; ++a) {
//             for (size_t b = 0; b < 10; ++b) {
//                 for (size_t c = 0; c < 15; ++c) {
//                     array[a][b][c] = i;
//                     ++i;
//                 }
//             }
//         }
//     }
// }

TEST(kd_array, loebel_syntax) {
    kd_array<size_t, 3> array({5, 10, 15});

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    array[{a, b, c}] = i;
                    ++i;
                }
            }
        }
    }

    {
        size_t i = 0;
        for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 10; ++b) {
                for (size_t c = 0; c < 15; ++c) {
                    size_t v = array[{a, b, c}];
                    ASSERT_EQ(i, v);
                    ++i;
                }
            }
        }
    }
}
