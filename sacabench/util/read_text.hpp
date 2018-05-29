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
     * \brief Reads content of a txt file into a string.
     * \param filepath Path of the file to be read.
     */
    string read_text (std::string filepath) {

        std::vector<character> text;
        std::string line;
        std::ifstream filestream(filepath.c_str());

        // Make sure the file could be opened.
        if (!filestream) {
            std::cout << "Error! Incorrect file." << std::endl;
            exit(EXIT_FAILURE);
        }

        // Insert first line.
        std::getline(filestream, line);
        std::copy(line.begin(), line.end(), std::back_inserter(text));

        // Insert rest of lines together with a '\n'.
        while (!filestream.eof()) {
            text.push_back('\n');
            std::getline(filestream, line);
            // Insert chars of current line into string text.
            std::copy(line.begin(), line.end(), std::back_inserter(text));
        }

        // TODO: Instead of making a copy here, replace implementation above
        // with something that directly writes into a `util::string`.
        return make_string(span(text));
    }
}

/******************************************************************************/
