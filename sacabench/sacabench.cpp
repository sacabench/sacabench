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

    CLI::App app{"CLI for SACABench."};
    app.require_subcommand();
    app.set_failure_message(CLI::FailureMessage::help);

    CLI::App& list =
        *app.add_subcommand("list", "List all Implemented Algorithms.");
    bool no_desc;
    {
        list.add_flag("--no-description", no_desc,
                      "Don't show a description for each Algorithm.");
    }

    CLI::App& construct = *app.add_subcommand("construct", "Construct a SA.");
    std::string input_filename = "";
    std::string output_filename = "";
    std::string algorithm = "";
    bool check_sa;
    bool record_benchmark;
    {
        construct.add_option("-i,--in", input_filename, "Path to input file.")
            ->required()
            ->check(CLI::ExistingFile);
        construct
            .add_option("-o,--out", output_filename, "Path to output file.")
            ->check(CLI::NonexistentPath);
        construct
            .add_option("-a,--algorithm", algorithm, "Which Algorithm to run.")
            ->required();
        construct.add_flag("--check", check_sa, "Check the constructed SA.");
        construct.add_flag("-b,--benchmark", record_benchmark,
                           "Record benchmark and display as JSON.");
    }

    CLI11_PARSE(app, argc, argv);

    // Handle CLI arguments
    auto& saca_list = util::saca_list::get();

    auto implemented_algos = [&] {
        std::cout << "Currently implemented Algorithms:" << std::endl;
        for (const auto& a : saca_list) {
            std::cout << "  [" << a->name() << "]" << std::endl;
            if (!no_desc) {
                std::cout << "    " << a->description() << std::endl;
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    };

    if (list) {
        implemented_algos();
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
            std::cerr << "ERROR: Algorithm does not exist" << std::endl;
            no_desc = true;
            implemented_algos();
            return 1;
        }
        {
            tdc::StatPhase root("CLI");
            {
                auto text = util::text_initializer_from_file(input_filename);
                auto sa = algo->construct_sa(text);
                if (check_sa) {
                    tdc::StatPhase check_sa_phase("SA Checker");

                    // Read the string in again
                    auto s = util::string(text.text_size());
                    text.initializer(s);

                    // Run the SA checker, and print the result
                    auto res = sa->check(s);
                    if (res != util::sa_check_result::ok) {
                        std::cerr << "ERROR: SA check failed" << std::endl;
                        return 1;
                    } else {
                        std::cerr << "SA check OK" << std::endl;
                    }
                }
                root.log("algorithm_name", algo->name());
            }

            if (record_benchmark) {
                auto j = root.to_json();
                std::cout << j.dump(4) << std::endl;
            }
        }
    }

    return 0;
}

/******************************************************************************/
