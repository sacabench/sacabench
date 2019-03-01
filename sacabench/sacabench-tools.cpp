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
#include <thread>

#include <CLI/CLI.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sched.h>

std::int32_t main(std::int32_t argc, char const** argv) {
    CLI::App app{"CLI for SACABench support tools."};
    app.require_subcommand();
    app.failure_message(CLI::FailureMessage::help);

    CLI::App& chk_multi =
        *app.add_subcommand("mt", "Check multithreading behavior.");
    uint32_t num_threads;
    {
        chk_multi.add_option("num_threads", num_threads, "Number of threads.")
            ->required();
    }

    CLI11_PARSE(app, argc, argv);

    if (chk_multi) {
        std::cout << "CLI argument: " << num_threads << std::endl;
        std::cout << "omp_get_max_threads: " << omp_get_max_threads() << std::endl;
        std::cout << "std::thread::hardware_concurrency(): " << std::thread::hardware_concurrency() << std::endl;
    }

    return 0;
}
