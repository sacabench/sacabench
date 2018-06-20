/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#include <CLI/CLI.hpp>

#include "util/bucket_size.hpp"
#include "util/container.hpp"
#include "util/saca.hpp"

bool file_exist_check(std::string const& path) {
    std::ifstream f(path.c_str());
    bool res = f.good();
    return res;
}

std::int32_t main(std::int32_t argc, char const** argv) {
    int late_fail = 0;
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
    std::string benchmark_filename = "";
    bool check_sa = false;
    bool out_json = false;
    bool out_binary = false;
    uint8_t out_fixed_bits = 0;
    bool force_overwrite = false;
    {
        construct.add_option("algorithm", algorithm, "Which Algorithm to run.")
            ->required();
        construct
            .add_option("input", input_filename,
                        "Path to input file, or - for STDIN.")
            ->required();
        auto opt_output =
            construct.add_option("-o,--output", output_filename,
                                 "Path to SA output file, or - for STDOUT.");
        construct.add_flag("-c,--check", check_sa, "Check the constructed SA.");
        construct.add_option("-b,--benchmark", benchmark_filename,
                             "Record benchmark and output as JSON. Takes Path "
                             "to output file, or - for STDOUT");

        auto opt_json = construct.add_flag("-J,--json", out_json,
                                           "Output SA as JSON array.");
        auto opt_binary = construct.add_flag(
            "-B,--binary", out_binary,
            "Output SA as binary array of unsigned integers, with a 1 Byte "
            "header "
            "describing the number of bits used for each integer.");

        opt_json->needs(opt_output);
        opt_json->excludes(opt_binary);

        opt_binary->needs(opt_output);
        opt_binary->excludes(opt_json);

        auto opt_fixed_bits = construct.add_option(
            "-F,--fixed", out_fixed_bits,
            "Elide the header, and output a fixed number of bits per SA entry");

        opt_fixed_bits->needs(opt_binary);

        construct.add_flag(
            "-f,--force", force_overwrite,
            "Overwrite existing Files instead of raising an error.");
    }

    CLI::App& demo =
        *app.add_subcommand("demo", "Run all Algorithms on an example string.");

    CLI::App& batch =
        *app.add_subcommand("batch", "Measure runtime and memory usage for all algorithms.");
    {
        batch
            .add_option("input", input_filename,
                        "Path to input file, or - for STDIN.")
            ->required();
        batch.add_flag("-c,--check", check_sa, "Check the constructed SA.");
        batch.add_option("-b,--benchmark", benchmark_filename,
                             "Record benchmark and output as JSON. Takes Path "
                             "to output file, or - for STDOUT");
        batch.add_flag(
            "-f,--force", force_overwrite,
            "Overwrite existing Files instead of raising an error.");
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
                std::unique_ptr<util::text_initializer> text;
                std::string stdin_buf;

                if (input_filename == "-") {
                    stdin_buf = std::string(
                        std::istreambuf_iterator<char>(std::cin), {});
                    text = std::make_unique<util::text_initializer_from_span>(
                        util::string_span(
                            (util::character const*)stdin_buf.data(),
                            stdin_buf.size()));
                } else {
                    if (!file_exist_check(input_filename)) {
                        std::cerr << "ERROR: Input File " << input_filename
                                  << " does not exist." << std::endl;
                        return 1;
                    }

                    text = std::make_unique<util::text_initializer_from_file>(
                        input_filename);
                }

                auto sa = algo->construct_sa(*text);

                if (output_filename.size() != 0) {
                    tdc::StatPhase check_sa_phase("Output SA");

                    auto write_out = [&](std::ostream& out_stream) {
                        if (out_json) {
                            sa->write_json(out_stream);
                        }
                        if (out_binary) {
                            sa->write_binary(out_stream, out_fixed_bits);
                        }
                    };

                    if (output_filename == "-") {
                        write_out(std::cout);
                    } else {
                        if (!force_overwrite &&
                            file_exist_check(output_filename)) {
                            std::cerr << "ERROR: Output File "
                                      << output_filename
                                      << " does already exist." << std::endl;
                            return 1;
                        }
                        std::ofstream out_file(output_filename,
                                               std::ios_base::out |
                                                   std::ios_base::binary |
                                                   std::ios_base::trunc);
                        write_out(out_file);
                    }
                }
                if (check_sa) {
                    tdc::StatPhase check_sa_phase("SA Checker");

                    // Read the string in again
                    auto s = util::string(text->text_size());
                    text->initializer(s);

                    // Run the SA checker, and print the result
                    auto res = sa->check(s);
                    check_sa_phase.log("check_result", res);
                    if (res != util::sa_check_result::ok) {
                        std::cerr << "ERROR: SA check failed" << std::endl;
                        late_fail = 1;
                    } else {
                        std::cerr << "SA check OK" << std::endl;
                    }
                }
                root.log("algorithm_name", algo->name());
            }

            if (benchmark_filename.size() > 0) {
                auto write_bench = [&](std::ostream& out) {
                    auto j = root.to_json();
                    out << j.dump(4) << std::endl;
                };

                if (benchmark_filename == "-") {
                    write_bench(std::cout);
                } else {
                    if (!force_overwrite &&
                        file_exist_check(benchmark_filename)) {
                        std::cerr << "ERROR: Benchmark File "
                                  << benchmark_filename
                                  << " does already exist." << std::endl;
                        return 1;
                    }
                    std::ofstream benchmark_file(benchmark_filename,
                                                 std::ios_base::out |
                                                     std::ios_base::binary |
                                                     std::ios_base::trunc);
                    write_bench(benchmark_file);
                }
            }
        }
    }

    if (demo) {
        for (const auto& a : saca_list) {
            std::cout << "Running " << a->name() << "..." << std::endl;
            a->run_example();
        }
    }

    if (batch) {
        std::cout << "Loading input..." << std::endl;
        std::unique_ptr<util::text_initializer> text;
        std::string stdin_buf;

        if (input_filename == "-") {
            stdin_buf = std::string(
                    std::istreambuf_iterator<char>(std::cin), {});
            text = std::make_unique<util::text_initializer_from_span>(
                    util::string_span(
                        (util::character const*)stdin_buf.data(),
                        stdin_buf.size()));
        } else {
            if (!file_exist_check(input_filename)) {
                std::cerr << "ERROR: Input File " << input_filename
                    << " does not exist." << std::endl;
                return 1;
            }

            text = std::make_unique<util::text_initializer_from_file>(
                    input_filename);
        }

        nlohmann::json stat_array = nlohmann::json::array();

        for (const auto& algo : saca_list) {
            tdc::StatPhase root(algo->name().data());
            {
                std::cout << "Running " << algo->name() << "..." << std::endl;
                auto sa = algo->construct_sa(*text);

                if (check_sa) {
                    tdc::StatPhase check_sa_phase("SA Checker");

                    // Read the string in again
                    auto s = util::string(text->text_size());
                    text->initializer(s);

                    // Run the SA checker, and print the result
                    auto res = sa->check(s);
                    check_sa_phase.log("check_result", res);
                    if (res != util::sa_check_result::ok) {
                        std::cerr << "ERROR: SA check failed" << std::endl;
                        late_fail = 1;
                    } else {
                        std::cerr << "SA check OK" << std::endl;
                    }
                }
                root.log("algorithm_name", algo->name());
            }
            stat_array.push_back(root.to_json());
        }

        if (benchmark_filename.size() > 0) {
            auto write_bench = [&](std::ostream& out) {
                //auto j = stat_array.to_json();
                out << stat_array.dump(4) << std::endl;
            };

            if (benchmark_filename == "-") {
                write_bench(std::cout);
            } else {
                if (!force_overwrite &&
                        file_exist_check(benchmark_filename)) {
                    std::cerr << "ERROR: Benchmark File "
                        << benchmark_filename
                        << " does already exist." << std::endl;
                    return 1;
                }
                std::ofstream benchmark_file(benchmark_filename,
                        std::ios_base::out |
                        std::ios_base::binary |
                        std::ios_base::trunc);
                write_bench(benchmark_file);
            }
        }
    }

    return late_fail;
}

/******************************************************************************/
