/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <cstdint>
#include <CLI/CLI.hpp>
#include "util/bucket_size.hpp"
#include "util/container.hpp"

#include "util/saca.hpp"

std::int32_t main(std::int32_t argc, char const** argv) {

    CLI::App app{"App description"};

    std::string input_filename = "";
    app.add_option("-f,--file", input_filename, "Path to input file");

    app.add_flag("-l,--list", "Show a list of available algorithms");

    CLI11_PARSE(app, argc, argv);

    //std::cout << "Input file: " << (input_filename == "") ? "- not set -" : input_filename << std::endl;
    //std::cout << "Show list: " << (app.count("--list") > 0) ? "true" : "false" << std::endl;

    /*
    auto& saca_list = sacabench::util::saca_list::get();
    for (const auto& a : saca_list) {
        std::cout << "Running " << a->name() << "..." << std::endl;
        a->run_example();
    }
    */

    return 0;
}

/******************************************************************************/
