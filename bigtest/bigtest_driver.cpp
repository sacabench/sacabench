/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "gtest/gtest.h"

int main(int argc, char** argv) {
#ifdef DEBUG
    // Run really big tests only in release mode
    std::cout << "Skipping really big tests because of Debug mode."
              << std::endl;
    std::cout << "Run `cmake -DCMAKE_BUILD_TYPE=Release ..` to switch to "
                 "Release build."
              << std::endl;
#endif
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
