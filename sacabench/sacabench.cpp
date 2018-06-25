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
        *app.add_subcommand("list", "List all implemented algorithms.");
    bool no_desc;
    {
        list.add_flag("--no-description", no_desc,
                      "Don't show a description for each algorithm.");
    }

    CLI::App& construct = *app.add_subcommand("construct", "Construct a SA.");
    std::string input_filename = "";
    std::string output_json_filename = "";
    std::string output_binary_filename = "";
    std::string algorithm = "";
    std::string benchmark_filename = "";
    bool check_sa = false;
    uint32_t out_fixed_bits = 0;
    std::string prefix_size = "";
    bool force_overwrite = false;
    uint32_t sa_minimum_bits = 32;
    uint32_t repetition_count = 1;
    bool plot = false;
    {
        construct.add_option("algorithm", algorithm, "Which algorithm to run.")
            ->required();
        construct
            .add_option("input", input_filename,
                        "Path to input file, or - for STDIN.")
            ->required();
        construct.add_flag("-c,--check", check_sa, "Check the constructed SA.");
        construct.add_option("-b,--benchmark", benchmark_filename,
                             "Record benchmark and output as JSON. Takes path "
                             "to output file, or - for STDOUT");

        construct.add_option("-J,--json", output_json_filename,
                             "Output SA as JSON array. Takes path to output "
                             "file, or - for STDOUT.");
        auto opt_binary = construct.add_option(
            "-B,--binary", output_binary_filename,
            "Output SA as binary array of unsigned integers, with a 1 Byte "
            "header "
            "describing the number of bits used for each integer. Takes path "
            "to output file, or - for STDOUT.");

        auto opt_fixed_bits = construct.add_option(
            "-F,--fixed", out_fixed_bits,
            "Elide the header, and output a fixed number of "
            "bits per SA entry");

        opt_fixed_bits->needs(opt_binary);

        construct.add_option("-p,--prefix", prefix_size,
                             "calculate SA of prefix of input.");

        construct.add_flag(
            "-f,--force", force_overwrite,
            "Overwrite existing files instead of raising an error.");

        construct.add_option(
            "-m,--minimum_sa_bits", sa_minimum_bits,
            "The lower bound of bits to use per SA entry during "
            "construction",
            32);
        construct.add_option(
            "-r,--repetitions", repetition_count,
            "The value indicates the number of times the SACA(s) will run. A "
            "larger number will possibly yield more accurate results",
            1);
        
        construct.add_flag("-z,--plot", plot, "Plot measurements.");
    }

    CLI::App& demo =
        *app.add_subcommand("demo", "Run all algorithms on an example string.");

    CLI::App& batch = *app.add_subcommand(
        "batch", "Measure runtime and memory usage for all algorithms.");
    {
        batch
            .add_option("input", input_filename,
                        "Path to input file, or - for STDIN.")
            ->required();
        batch.add_flag("-c,--check", check_sa, "Check the constructed SA.");
        batch.add_option("-b,--benchmark", benchmark_filename,
                         "Record benchmark and output as JSON. Takes path "
                         "to output file, or - for STDOUT");
        batch.add_flag("-f,--force", force_overwrite,
                       "Overwrite existing files instead of raising an error.");
        batch.add_option("-m,--minimum_sa_bits", sa_minimum_bits,
                         "The lower bound of bits to use per SA entry during "
                         "construction",
                         32);
        batch.add_option("-p,--prefix", prefix_size,
                         "calculate SA of prefix of input.");
        batch.add_option(
            "-r,--repetitions", repetition_count,
            "The value indicates the number of times the SACA(s) will run. A "
            "larger number will possibly yield more accurate results",
            1);
        batch.add_flag("-z,--plot", plot, "Plot measurements.");
    }

    CLI11_PARSE(app, argc, argv);

    // Check early if file exists
    bool out_json = output_json_filename.size() != 0;
    bool out_binary = output_binary_filename.size() != 0;
    bool out_benchmark = benchmark_filename.size() != 0;
    auto check_out_filename = [&](std::string const& filename,
                                  std::string const& name) {
        if (!force_overwrite && filename.size() != 0 && filename != "-" &&
            file_exist_check(filename)) {
            std::cerr << "ERROR: " << name << " file " << filename
                      << " does already exist." << std::endl;
            return true;
        }
        return false;
    };
    auto check_in_filename = [&](std::string const& filename,
                                 std::string const& name) {
        if (filename.size() != 0 && filename != "-" &&
            !file_exist_check(filename)) {
            std::cerr << "ERROR: " << name << " file " << filename
                      << " does not exist." << std::endl;
            return true;
        }
        return false;
    };

    // Handle CLI arguments
    auto& saca_list = util::saca_list::get();

    auto implemented_algos = [&] {
        std::cout << "Currently implemented algorithms:" << std::endl;
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
        if (check_out_filename(output_json_filename, "JSON output")) {
            return 1;
        }
        if (check_out_filename(output_binary_filename, "Binary output")) {
            return 1;
        }
        if (check_out_filename(benchmark_filename, "Benchmark")) {
            return 1;
        }
        if (check_in_filename(input_filename, "Input")) {
            return 1;
        }
        {
            nlohmann::json sum_array = nlohmann::json::array();
            for (uint32_t i = 0; i < repetition_count; i++) {
                tdc::StatPhase root("CLI");
                {
                    std::unique_ptr<util::text_initializer> text;
                    std::string stdin_buf;
                    size_t prefix = -1;

                    if (prefix_size.size() > 0) {
                        try {
                            uint32_t unit_factor;
                            size_t input_prefix;
                            if (prefix_size[prefix_size.size() - 1] == 'K') {
                                std::string number_part = prefix_size.substr(
                                    0, prefix_size.size() - 1);
                                input_prefix = std::stoi(number_part);
                                unit_factor = 1024;
                            } else if (prefix_size[prefix_size.size() - 1] ==
                                       'M') {
                                std::string number_part = prefix_size.substr(
                                    0, prefix_size.size() - 1);
                                input_prefix = std::stoi(number_part);
                                unit_factor = 1024 * 1024;
                            } else {
                                std::string number_part =
                                    prefix_size.substr(0, prefix_size.size());
                                input_prefix = std::stoi(number_part);
                                unit_factor = 1;
                            }
                            prefix = input_prefix * unit_factor;
                        } catch (const std::invalid_argument& ia) {
                            std::cerr << "ERROR: input prefix is not a "
                                         "valid prefix value."
                                      << std::endl;
                            return 1;
                        }
                    }

                    if (input_filename == "-") {
                        stdin_buf = std::string(
                            std::istreambuf_iterator<char>(std::cin), {});
                        text =
                            std::make_unique<util::text_initializer_from_span>(
                                util::string_span(
                                    (util::character const*)stdin_buf.data(),
                                    stdin_buf.size()),
                                prefix);
                    } else {
                        text =
                            std::make_unique<util::text_initializer_from_file>(
                                input_filename, prefix);
                    }

                    auto sa = algo->construct_sa(*text, sa_minimum_bits);

                    if (out_json | out_binary) {
                        tdc::StatPhase check_sa_phase("Output SA");

                        auto handle_output_opt = [&](std::string const& opt,
                                                     auto write_out) {
                            if (opt.size() == 0) {
                                return;
                            }

                            if (opt == "-") {
                                write_out(std::cout);
                            } else {
                                std::ofstream out_file(
                                    opt, std::ios_base::out |
                                             std::ios_base::binary |
                                             std::ios_base::trunc);
                                write_out(out_file);
                            }
                        };

                        handle_output_opt(output_json_filename,
                                          [&](std::ostream& stream) {
                                              sa->write_json(stream);
                                          });

                        handle_output_opt(
                            output_binary_filename, [&](std::ostream& stream) {
                                sa->write_binary(stream, out_fixed_bits);
                            });
                    }
                    if (check_sa) {
                        tdc::StatPhase check_sa_phase("SA Checker");

                        // Read the string in again
                        size_t text_size = text->text_size();
                        auto s = util::string(text_size);
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
                }
                root.log("algorithm_name", algo->name());
                sum_array.push_back(root.to_json());
            }

            if (out_benchmark) {
                auto write_bench = [&](std::ostream& out) {
                    // auto j = root.to_json();
                    auto j = sum_array;
                    out << j.dump(4) << std::endl;
                };

                if (benchmark_filename == "-") {
                    write_bench(std::cout);
                } else {
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
            std::cerr << "Running " << a->name() << "..." << std::endl;
            a->run_example();
        }
    }

    if (batch) {
        if (check_out_filename(benchmark_filename, "Benchmark")) {
            return 1;
        }
        if (check_in_filename(input_filename, "Input")) {
            return 1;
        }

        std::cerr << "Loading input..." << std::endl;
        std::unique_ptr<util::text_initializer> text;
        std::string stdin_buf;
        size_t prefix = -1;

        if (prefix_size.size() > 0) {
            try {
                uint32_t unit_factor;
                size_t input_prefix;
                if (prefix_size[prefix_size.size() - 1] == 'K') {
                    std::string number_part =
                        prefix_size.substr(0, prefix_size.size() - 1);
                    input_prefix = std::stoi(number_part);
                    unit_factor = 1024;
                } else if (prefix_size[prefix_size.size() - 1] == 'M') {
                    std::string number_part =
                        prefix_size.substr(0, prefix_size.size() - 1);
                    input_prefix = std::stoi(number_part);
                    unit_factor = 1024 * 1024;
                } else {
                    std::string number_part =
                        prefix_size.substr(0, prefix_size.size());
                    input_prefix = std::stoi(number_part);
                    unit_factor = 1;
                }
                prefix = input_prefix * unit_factor;
            } catch (const std::invalid_argument& ia) {
                std::cerr << "ERROR: input prefix is not a "
                             "valid prefix value."
                          << std::endl;
                return 1;
            }
        }

        if (input_filename == "-") {
            stdin_buf =
                std::string(std::istreambuf_iterator<char>(std::cin), {});
            text = std::make_unique<util::text_initializer_from_span>(
                util::string_span((util::character const*)stdin_buf.data(),
                                  stdin_buf.size()),
                prefix);
        } else {
            text = std::make_unique<util::text_initializer_from_file>(
                input_filename, prefix);
        }

        nlohmann::json stat_array = nlohmann::json::array();

        for (const auto& algo : saca_list) {
            nlohmann::json alg_array = nlohmann::json::array();

            for (uint32_t i = 0; i < repetition_count; i++) {
                tdc::StatPhase root(algo->name().data());
                {
                    std::cerr << "Running " << algo->name() << " (" << (i + 1)
                              << "/" << repetition_count << ")" << std::endl;

                    auto sa = algo->construct_sa(*text, sa_minimum_bits);

                    if (check_sa) {
                        tdc::StatPhase check_sa_phase("SA Checker");

                        // Read the string in again
                        size_t text_size = text->text_size();
                        auto s = util::string(text_size);
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
                }
                root.log("algorithm_name", algo->name());
                alg_array.push_back(root.to_json());
            }
            stat_array.push_back(alg_array);
        }

        if (out_benchmark) {
            auto write_bench = [&](std::ostream& out) {
                // auto j = stat_array.to_json();
                out << stat_array.dump(4) << std::endl;
            };

            if (benchmark_filename == "-") {
                write_bench(std::cout);
            } else {
                std::ofstream benchmark_file(benchmark_filename,
                                             std::ios_base::out |
                                                 std::ios_base::binary |
                                                 std::ios_base::trunc);
                write_bench(benchmark_file);
            }
        }
    }

    if(plot && out_benchmark){
            std::string r_command = "R CMD BATCH --no-save --no-restore '--args " + benchmark_filename + "'  ..//stats/stat_plot.R test.Rout"; 
            int i = system(r_command.c_str());
            (void)i; //suppress  warning
    }
        
    return late_fail;
}

/******************************************************************************/
