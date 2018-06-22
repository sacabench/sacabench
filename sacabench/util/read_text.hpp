/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include "util/string.hpp"

namespace sacabench::util {

    /**
     * This struct contains the size of the text and provides a function to read the text file into a string.
     */
    struct read_text_context {
        mutable std::ifstream file;
        size_t size = 0;

        read_text_context() = default;
        read_text_context(read_text_context&& other) = default;
        read_text_context& operator=(read_text_context&& other) = default;

        read_text_context(std::string filepath) {
            file.open(filepath, std::ios::in|std::ios::binary);
            file.seekg(0, std::ios::end); // set the pointer to the end
            size = file.tellg() ; // get the length of the file
        }

        /**
         * \brief Reads content of a txt file into a string.
         * This function reads the content of a text file at the given path bytewise.
         */
        inline void read_text(span<character> out_text) const {
            //DCHECK_EQ(out_text.size(), this->size);

            file.seekg(0, std::ios::beg); // set the pointer to the beginning
            file.read( (char*) out_text.data(), out_text.size());
        }
    };
}

/******************************************************************************/
