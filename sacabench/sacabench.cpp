/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

#include <CLI/CLI.hpp>

#include "util/bucket_size.hpp"
#include "util/container.hpp"
#include "util/saca.hpp"

#include "source_dir_constant.hpp"

#include <stdio.h>
#include <stdlib.h>

bool file_exist_check(std::string const& path) {
    std::ifstream f(path.c_str());
    bool res = f.good();
    return res;
}

nlohmann::json load_json(std::string const& path) {
    std::ifstream f(path.c_str());
    nlohmann::json j;
    f >> j;
    return j;
}

struct output_t {
    std::string output;
    int exitcode;

    operator bool() const { return exitcode == 0; }
};

output_t get_output_from_cmd(std::string cmd) {
    std::string data;
    std::FILE* stream;
    const int max_buffer = 256;
    char buffer[max_buffer];
    cmd.append(" 2>&1");
    int exitcode = 101;

    stream = popen(cmd.c_str(), "r");
    if (stream) {
        while (!std::feof(stream)) {
            if (std::fgets(buffer, max_buffer, stream) != NULL) {
                data.append(buffer);
            }
        }
        int st = pclose(stream);
        if (WIFEXITED(st)) {
            exitcode = WEXITSTATUS(st);
        }
    }
    return output_t{
        data,
        exitcode,
    };
}

int get_exit_code_from_cmd(std::string cmd) {
    int status = system(cmd.c_str());
    if (status < 0) {
        std::cerr << "ERROR: " << strerror(errno) << '\n';
        return 102; // system() call error
    } else {
        if (WIFEXITED(status))
            return WEXITSTATUS(status);
        else {
            return 101; // abnormal exit
        }
    }
}

std::string get_short_filename(std::string const& s) {
    auto filename_start_index = s.find_last_of("\\/");
    if (filename_start_index != std::string::npos) {
        filename_start_index += 1;
        return s.substr(filename_start_index);
    } else {
        return s;
    }
}

std::string get_parent_path(std::string const& s) {
    return s.substr(0, s.find_last_of("\\/"));
}

void remove_newline(std::string& s) {
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
}

int parse_prefix_if_set(std::string const& prefix_size, size_t& prefix) {
    if (prefix_size.size() > 0) {
        try {
            size_t unit_factor = 0;
            size_t input_prefix = 0;
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
    return 0;
}

struct sysinfo_t {
    std::string all_lscpu_info;
    std::string all_uname_info;
    std::string all_mem_info;
};
sysinfo_t get_full_sysinfo() {
    sysinfo_t r;
    auto cmd_all_lscpu_info = "lscpu";
    auto cmd_all_uname_info = "uname -a";
    auto cmd_all_mem_info = "free -t";

    r.all_lscpu_info = get_output_from_cmd(cmd_all_lscpu_info).output;
    r.all_uname_info = get_output_from_cmd(cmd_all_uname_info).output;
    r.all_mem_info = get_output_from_cmd(cmd_all_mem_info).output;

    return r;
}

nlohmann::json get_config_json(size_t prefix, size_t repetition_count,
                               std::string input_filename) {
    // Create json file with config
    // file name, prefix size, amount of repetitions
    nlohmann::json config_json = nlohmann::json::array();

    auto cmd_operating_system = "uname";
    auto cmd_model_name =
        "lscpu | grep 'Model name' | cut -f 2 -d ':'| awk '{$1=$1}1'";
    auto cmd_amount_cpus = "grep -c ^processor /proc/cpuinfo";
    auto cmd_threads_per_socket =
        "lscpu | grep 'Thread(s)' | cut -f 2 -d ':'| awk '{$1=$1}1'";

    auto operating_system = get_output_from_cmd(cmd_operating_system).output;
    remove_newline(operating_system);

    auto model_name = get_output_from_cmd(cmd_model_name).output;
    remove_newline(model_name);

    auto amount_cpus = get_output_from_cmd(cmd_amount_cpus).output;
    remove_newline(amount_cpus);

    auto threads_per_socket =
        get_output_from_cmd(cmd_threads_per_socket).output;
    remove_newline(threads_per_socket);

    // input_filename contains full path to input file. For config_json file we
    // only need the name.
    input_filename = get_short_filename(input_filename);

    auto sysinfo = get_full_sysinfo();

    nlohmann::json j = {
        {"input", input_filename},
        {"prefix", prefix},
        {"repetitions", repetition_count},
        {"operating_system", operating_system},
        {"model_name", model_name},
        {"amount_cpus", amount_cpus},
        {"threads_per_socket", threads_per_socket},
        {"all_lscpu_info", sysinfo.all_lscpu_info},
        {"all_uname_info", sysinfo.all_uname_info},
        {"all_mem_info", sysinfo.all_mem_info},
    };

    config_json.push_back(j);
    return config_json;
}

struct benchmark_json_info {
    enum benchmark_json_format {
        unknown,
        construct,
        batch,
    };

    benchmark_json_format kind;
    std::string input_file;
    size_t prefix = 0;

    bool is_batch() { return kind == benchmark_json_format::batch; }
};

benchmark_json_info check_benchmark_json_format(nlohmann::json const& j) {
    auto r = benchmark_json_info::unknown;

    nlohmann::json const* first_entry = nullptr;
    if (j.is_array()) {
        r = benchmark_json_info::construct;
        if (j.size() > 0) {
            if (j.at(0).is_array()) {
                r = benchmark_json_info::batch;
                if (j.at(0).size() > 0) {
                    first_entry = &j.at(0).at(0);
                }
            } else {
                first_entry = &j.at(0);
            }
        }
    }

    auto r2 = benchmark_json_info{};
    r2.kind = r;

    if (first_entry) {
        auto& stats = (*first_entry)["stats"];
        for (auto& e : stats) {
            if (e["key"] == "input_file") {
                r2.input_file = e["value"];
            }
            if (e["key"] == "prefix") {
                std::string prefix_size = e["value"];
                parse_prefix_if_set(prefix_size, r2.prefix);
            }
        }
    }

    return r2;
}

benchmark_json_info load_benchmark_json_format(std::string const& path) {
    auto stats_json = load_json(path);
    return check_benchmark_json_format(stats_json);
}

void do_plot(std::string const& benchmark_filename, bool out_benchmark) {
    auto stats = load_benchmark_json_format(benchmark_filename);

    bool batch = stats.is_batch();
    auto short_input_filename = stats.input_file;
    size_t text_size = stats.prefix;

    std::stringstream r_command;
    r_command << "R CMD BATCH --no-save --no-restore '--args "
              << benchmark_filename;
    std::cerr << "plot benchmark...";
    if (batch) {
        r_command << " 1 " << short_input_filename << " " << text_size << "'  "
                  << SACABENCH_SOURCE_DIRECTORY << "/stats/stat_plot.R "
                  << benchmark_filename << ".Rout";
    } else if (out_benchmark) {
        r_command << " 0 " << short_input_filename << " " << text_size << "'  "
                  << SACABENCH_SOURCE_DIRECTORY << "/stats/stat_plot.R "
                  << benchmark_filename << ".Rout";
    } else {
        std::cerr << "not able to plot." << std::endl;
        return;
    }
    int exit_status = get_exit_code_from_cmd(r_command.str());
    if (exit_status != 0) {
        std::cerr << "error thrown while running R-script." << std::endl;
    } else {
        std::cerr << "saved as: " << benchmark_filename << ".pdf" << std::endl;
    }
}

std::unique_ptr<sacabench::util::text_initializer>
load_input(std::string const& input_filename, std::string const& prefix_size,
           std::string& stdin_buf) {
    using namespace sacabench;

    std::unique_ptr<util::text_initializer> text;

    size_t prefix = -1;
    if (parse_prefix_if_set(prefix_size, prefix)) {
        return nullptr;
    }

    if (input_filename == "-") {
        stdin_buf = std::string(std::istreambuf_iterator<char>(std::cin), {});
        text = std::make_unique<util::text_initializer_from_span>(
            util::string_span((util::character const*)stdin_buf.data(),
                              stdin_buf.size()),
            prefix);
    } else {
        text = std::make_unique<util::text_initializer_from_file>(
            input_filename, prefix);
    }

    return text;
}

std::int32_t main(std::int32_t argc, char const** argv) {
    int late_fail = 0;
    using namespace sacabench;

    CLI::App app{"CLI for SACABench."};
    app.require_subcommand();
    app.failure_message(CLI::FailureMessage::help);

    CLI::App& list =
        *app.add_subcommand("list", "List all implemented algorithms.");
    bool no_desc;
    bool list_json;
    {
        list.add_flag("-n,--no-description", no_desc,
                      "Don't show a description for each algorithm.");
        list.add_flag("-j,--json", list_json, "Output list as an json array");
    }

    CLI::App& construct = *app.add_subcommand("construct", "Construct a SA.");
    std::string input_filename = "";
    std::string output_json_filename = "";
    std::string output_binary_filename = "";
    std::string algorithm = "";
    std::string benchmark_filename = "";
    std::vector<std::string> whitelist = {};
    std::vector<std::string> blacklist = {};
    bool check_sa = false;
    uint32_t out_fixed_bits = 0;
    std::string prefix_size = "";
    bool force_overwrite = false;
    uint32_t sa_minimum_bits = 32;
    uint32_t repetition_count = 1;
    bool rplot = false;
    bool latexplot = false;
    bool fast_check = false;
    bool sysinfo = false;
    {
        construct.set_config("--config", "",
                             "Read an config file for CLI args");
        construct.add_option("algorithm", algorithm, "Which algorithm to run.")
            ->required();
        construct
            .add_option("input", input_filename,
                        "Path to input file, or - for STDIN.")
            ->required();
        construct.add_flag("-c,--check", check_sa, "Check the constructed SA.");
        construct.add_flag(
            "-q,--fastcheck", fast_check,
            "Check the constructed SA with a faster, parallel algorithm.");
        auto b_opt = construct.add_option(
            "-b,--benchmark", benchmark_filename,
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
                             "Calculate SA of prefix of size TEXT.");

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
        construct.add_flag("-z,--rplot", rplot, "Plots measurements with R.")
            ->needs(b_opt);
        construct
            .add_flag("--latexplot", latexplot,
                      "Plots measurements with LaTex and SqlPlotTools.")
            ->needs(b_opt);
        construct
            .add_flag("-s,--sysinfo", sysinfo,
                      "Add system information to benchmark output.")
            ->needs(b_opt);
    }

    CLI::App& demo =
        *app.add_subcommand("demo", "Run all algorithms on an example string.");

    CLI::App& batch = *app.add_subcommand(
        "batch", "Measure runtime and memory usage for all algorithms.");
    {
        batch.set_config("--config", "", "Read an config file for CLI args");

        batch
            .add_option("input", input_filename,
                        "Path to input file, or - for STDIN.")
            ->required();
        batch.add_flag("-c,--check", check_sa, "Check the constructed SA.");
        batch.add_flag(
            "-q,--fastcheck", fast_check,
            "Check the constructed SA with a faster, parallel algorithm.");
        auto b_opt =
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
        CLI::Option* wlist = batch.add_option(
            "--whitelist", whitelist, "Execute only specific algorithms");
        batch
            .add_option("--blacklist", blacklist,
                        "Blacklist algorithms from execution")
            ->excludes(wlist);
        batch.add_flag("-z,--rplot", rplot, "Plots measurements with R.")
            ->needs(b_opt);
        batch
            .add_flag("--latexplot", latexplot,
                      "Plots measurements with LaTex and SqlPlotTools.")
            ->needs(b_opt);
        batch
            .add_flag("-s,--sysinfo", sysinfo,
                      "Add system information to benchmark output.")
            ->needs(b_opt);
    }

    CLI::App& plot_app = *app.add_subcommand("plot", "Plot measurements.");
    {
        plot_app
            .add_option("benchmark_file", benchmark_filename,
                        "Path to benchmark json file.")
            ->required();
    }

    CLI::App& histogram = *app.add_subcommand(
        "histogram", "Compute the histogram of a input file.");
    bool no_hist = false;
    {
        histogram
            .add_option("input", input_filename,
                        "Path to input file, or - for STDIN.")
            ->required();
        histogram.add_option("-p,--prefix", prefix_size,
                             "Only use a prefix of the input.");
        histogram.add_flag("-n,--no-hist", no_hist,
                           "Only output alphabet size.");
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
        bool first = true;
        if (list_json) {
            std::cout << "[" << std::endl;
        }
        for (const auto& a : saca_list) {
            if (list_json) {
                if (first) {
                    std::cout << "\t\"" << a->name() << "\"";
                    first = false;
                } else {
                    std::cout << "," << std::endl
                              << "\t\"" << a->name() << "\"";
                }
            } else {
                std::cout << "  [" << a->name() << "]" << std::endl;
                if (!no_desc) {
                    std::cout << "    " << a->description() << std::endl;
                    std::cout << std::endl;
                }
            }
        }
        if (list_json) {
            std::cout << "\n]" << std::endl;
        }
        std::cout << std::endl;
    };

    auto maybe_do_latexplot = [&](size_t prefix, size_t repetition_count) {
        if (latexplot) {
            // Create json file with config
            // file name, prefix size, amount of repetitions
            nlohmann::json config_json =
                get_config_json(prefix, repetition_count, input_filename);

            auto write_config = [&](std::ostream& out) {
                out << config_json.dump(4) << std::endl;
            };

            std::ofstream config_file(std::string(SACABENCH_SOURCE_DIRECTORY) +
                                          "/zbmessung/sqlplot/plotconfig.json",
                                      std::ios_base::out |
                                          std::ios_base::binary |
                                          std::ios_base::trunc);
            write_config(config_file);

            std::string pdf_destination = get_parent_path(benchmark_filename);
            std::string command = std::string("sh -c ") +
                                  SACABENCH_SOURCE_DIRECTORY +
                                  "/zbmessung/automation.sh " +
                                  benchmark_filename + " " + pdf_destination;
            std::cout << command << std::endl;

            int exit_status = get_exit_code_from_cmd(command);
            if (exit_status != 0) {
                std::cerr
                    << "error thrown while running plot automation script."
                    << std::endl;
            }
        }
    };

    auto maybe_do_sa_check = [&](util::text_initializer const& text,
                                 util::abstract_sa const& sa) {
        if (check_sa || fast_check) {
            tdc::StatPhase check_sa_phase("SA Checker");

            // Read the string in again
            size_t text_size = text.text_size();
            auto s = util::string(text_size);
            text.initializer(s);

            // Run the SA checker, and print the result
            auto res = sa.check(s, fast_check);
            check_sa_phase.log("check_result", res);
            if (res != util::sa_check_result::ok) {
                std::cerr << "ERROR: SA check failed" << std::endl;
                late_fail = 1;
            } else {
                std::cerr << "SA check OK" << std::endl;
            }
        }
    };

    auto maybe_do_output_benchmark = [&](nlohmann::json const& j) {
        if (out_benchmark) {
            auto write_bench = [&](std::ostream& out) {
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
    };

    sysinfo_t sysinfo_cache;

    auto log_root_stats = [&](tdc::StatPhase& root, util::saca const& algo,
                              size_t textsize) {
        auto short_input_filename = get_short_filename(input_filename);

        root.log("algorithm_name", algo.name());
        root.log("input_file", short_input_filename);
        root.log("repetitions", repetition_count);
        root.log("thread_count", omp_get_max_threads());
        root.log("prefix", textsize);

        if (sysinfo) {
            root.log("all_lscpu_info", sysinfo_cache.all_lscpu_info);
            root.log("all_mem_info", sysinfo_cache.all_mem_info);
            root.log("all_uname_info", sysinfo_cache.all_uname_info);
        }
    };

    if (sysinfo) {
        sysinfo_cache = get_full_sysinfo();
    }

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

        size_t automation_prefix = 0;
        {
            nlohmann::json sum_array = nlohmann::json::array();
            for (uint32_t i = 0; i < repetition_count; i++) {
                tdc::StatPhase root("CLI");
                {
                    std::string stdin_buf;
                    std::unique_ptr<util::text_initializer> text =
                        load_input(input_filename, prefix_size, stdin_buf);
                    if (text == nullptr) {
                        return 1;
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
                    maybe_do_sa_check(*text, *sa);

                    log_root_stats(root, *algo, text->text_size());
                    automation_prefix = text->text_size();
                }
                sum_array.push_back(root.to_json());
            }

            maybe_do_output_benchmark(sum_array);
        }

        maybe_do_latexplot(automation_prefix, repetition_count);
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
        std::string stdin_buf;
        std::unique_ptr<util::text_initializer> text =
            load_input(input_filename, prefix_size, stdin_buf);
        if (text == nullptr) {
            return 1;
        }

        nlohmann::json stat_array = nlohmann::json::array();

        size_t sanity_counter = 0;
        for (const auto& algo : saca_list) {
            if (!whitelist.empty()) {
                if (std::find(whitelist.begin(), whitelist.end(),
                              algo->name().data()) == whitelist.end()) {
                    continue;
                }
            }
            if (std::find(blacklist.begin(), blacklist.end(),
                          algo->name().data()) != blacklist.end()) {
                continue;
            }
            nlohmann::json alg_array = nlohmann::json::array();

            for (uint32_t i = 0; i < repetition_count; i++) {
                sanity_counter++;
                tdc::StatPhase root(algo->name().data());
                {
                    std::cerr << "Running " << algo->name() << " (" << (i + 1)
                              << "/" << repetition_count << ")" << std::endl;

                    auto sa = algo->construct_sa(*text, sa_minimum_bits);

                    maybe_do_sa_check(*text, *sa);

                    log_root_stats(root, *algo, text->text_size());
                }
                alg_array.push_back(root.to_json());
            }
            stat_array.push_back(alg_array);
        }

        maybe_do_output_benchmark(stat_array);
        maybe_do_latexplot(text->text_size(), repetition_count);

        if (sanity_counter == 0) {
            std::cerr << "ERROR: No Algorithm ran!\n";
            return 1;
        }
    }

    if (rplot || plot_app) {
        if (benchmark_filename == "-") {
            abort(); // TODO: this can not work!;
        }
        if (rplot) {
            do_plot(benchmark_filename, out_benchmark);
        }
        if (plot_app) {
            do_plot(benchmark_filename, true);
        }
    }

    if (histogram) {
        std::cout << "Loading input..." << std::endl;

        std::string stdin_buf;
        std::unique_ptr<util::text_initializer> text_init =
            load_input(input_filename, prefix_size, stdin_buf);
        if (text_init == nullptr) {
            return 1;
        }

        size_t const text_size = text_init->text_size();
        sacabench::util::string text(text_size);
        text_init->initializer(text.slice());

        std::cout << "Computing histogram for " << text_size
                  << " bytes of input..." << std::endl;

        std::array<size_t, 256> hist;
        for (auto& counter : hist) {
            counter = 0;
        }

        for (uint8_t byte : text) {
            hist[byte]++;
        }

        size_t alphabet_size = 0;
        for (auto& counter : hist) {
            alphabet_size += (counter != 0);
        }

        std::cout << "Alphabet size: " << alphabet_size << std::endl;

        if (!no_hist) {
            std::cout << "Histogram:" << std::endl;
            std::cout << "   hex, dec, chr,      count,  relative" << std::endl;
            for (size_t byte = 0; byte < 256; byte++) {
                char c = ' ';
                char b = ' ';
                if (std::isprint(char(byte))) {
                    c = byte;
                    b = '\'';
                }
                double percent = double(hist[byte]) / double(text_size) * 100.0;

                std::cout << "  "
                          << "0x" << std::setfill('0') << std::setw(2)
                          << std::hex << byte << ", " << std::setfill(' ')
                          << std::setw(3) << std::dec << byte << ", " << b << c
                          << b << ", " << std::setfill(' ') << std::setw(10)
                          << std::dec << hist[byte] << ", " << std::fixed
                          << std::setprecision(3) << std::setw(7) << percent
                          << " %" << std::endl;
            }
        }
    }

    return late_fail;
}

/******************************************************************************/
