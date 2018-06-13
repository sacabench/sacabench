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
    using namespace sacabench;

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

    bool check_sa;
    construct.add_flag("--check", check_sa);

    construct.add_flag("-b,--benchmark", "Record benchmark");

    CLI11_PARSE(app, argc, argv);

    // Handle CLI arguments
    auto& saca_list = util::saca_list::get();

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

    if (construct) {
        util::saca const* algo = nullptr;
        for (const auto& a : saca_list) {
            if (a->name() == algorithm) {
                algo = a;
                break;
            }
        }
        if (algo == nullptr) {
            std::cerr << "Algorithm does not exist" << std::endl;
            return 1;
        }

        auto text = util::text_initializer_from_file(input_filename);
        auto sa = algo->construct_sa(text);
        if (check_sa) {
            // Read the string in again
            auto s = util::string(text.text_size());
            text.initializer(s);

            // Run the SA checker, and print the result
            auto res = sa->check(s);
            if (res != util::sa_check_result::ok) {
                std::cerr << "SA check failed!" << std::endl;
                return 1;
            } else {
                std::cerr << "SA check OK." << std::endl;
                return 0;
            }
        }
    }

    return 0;
}

/******************************************************************************/
