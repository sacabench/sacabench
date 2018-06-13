/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "util/bucket_size.hpp"
#include "util/container.hpp"
#include <CLI/CLI.hpp>
#include <cstdint>

#include "util/saca.hpp"

std::int32_t main(std::int32_t argc, char const** argv) {

    CLI::App app{"App description"};
    app.require_subcommand();
    app.set_failure_message(CLI::FailureMessage::help);

    CLI::App& list = *app.add_subcommand("list", "TODO");
    bool no_desc;
    list.add_flag("--no-description", no_desc);
    CLI::App& construct = *app.add_subcommand("construct", "TODO");

    std::string input_filename = "";
    std::string output_filename = "";
    std::string algorithm = "";
    construct.add_option("-i,--in", input_filename, "Path to input file")
        ->required()
        ->check(CLI::ExistingFile);
    construct.add_option("-o,--out", output_filename, "Path to output file")
        ->check(CLI::NonexistentPath);
    construct.add_option("-a,--algorithm", algorithm, "Algorithm")->required();

    construct.add_flag("-b,--benchmark", "Record benchmark");

    CLI11_PARSE(app, argc, argv);

    // Handle CLI arguments
    auto& saca_list = sacabench::util::saca_list::get();

    if (list) {
        std::cout << "Currently implemented Algorithms:" << std::endl;
        for (const auto& a : saca_list) {
            std::cout << "  [" << a->name() << "]" << std::endl;
            if (!no_desc) {
                std::cout << "    " << a->description() << std::endl;
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    /*
    for (const auto& a : saca_list) {
        std::cout << "Running " << a->name() << "..." << std::endl;
        a->run_example();
    }
    */

    return 0;
}

/******************************************************************************/
