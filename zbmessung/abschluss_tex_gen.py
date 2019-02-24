#!/usr/bin/env python3

import subprocess
import pprint
import os
import argparse
import json
import statistics
import datetime
import sys

import tex_gen_module

def load_json_from_file(p):
    with open(str(p), 'r') as f:
        json_data = json.load(f)
    return json_data

def save_json_to_file(d, p):
    with open(str(p), 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)

def get_stats(stats):
    mp = {}
    for s in stats["stats"]:
        assert not s["key"] in mp
        mp[s["key"]] = s["value"]
    return mp

def find_subphase(phase, name):
    for n in phase["sub"]:
        if n["title"] == name:
            return n
    return None

def get_mem_time(phase):
    start = phase["timeStart"]
    end = phase["timeEnd"]
    peak = phase["memPeak"]

    duration = end - start

    return (peak, duration)

def handle_tablegen(args):
    path = args.path
    mode = args.mode

    outer_matrix = {}

    algorithms = set()
    files = set()
    threads_and_sizes = set()

    js = load_json_from_file(path)
    for repetitions in js:
        # Extract data from json collection
        assert len(repetitions) == 1
        root_phase = repetitions[0]
        checker_phase = find_subphase(root_phase, "SA Checker")
        saca_phase = find_subphase(root_phase, "SACA")
        algo_phase = find_subphase(saca_phase, "Algorithm")
        text_alloc_phase = find_subphase(saca_phase, "Allocate Text container")
        sa_alloc_phase = find_subphase(saca_phase, "Allocate SA container")

        root_stats = get_stats(root_phase)
        saca_stats = get_stats(saca_phase)
        check_stats = get_stats(checker_phase)

        input_file = root_stats["input_file"]
        prefix = int(root_stats["prefix"])
        assert int(root_stats["repetitions"]) == 1
        thread_count = int(root_stats["thread_count"])
        algorithm_name = root_stats["algorithm_name"]
        check_result = check_stats["check_result"]
        extra_sentinels = int(saca_stats["extra_sentinels"])
        sa_index_bit_size = int(saca_stats["sa_index_bit_size"])
        assert int(saca_stats["text_size"]) == prefix
        (algo_peak,
         algo_time) = get_mem_time(algo_phase)
        text_peak = get_mem_time(text_alloc_phase)[0]
        sa_peak = get_mem_time(sa_alloc_phase)[0]

        # Gather data in central data structure

        if (thread_count, prefix) not in outer_matrix:
            outer_matrix[(thread_count, prefix)] = {}

        matrix = outer_matrix[(thread_count, prefix)]

        if input_file not in matrix:
            matrix[input_file] = {}

        algorithms.add(algorithm_name)
        threads_and_sizes.add((thread_count, prefix))
        files.add(input_file)

        if algorithm_name not in matrix[input_file]:
            matrix[input_file][algorithm_name] = {
                "data" : "exists",
                "all" : [],
                "avg": {},
                "med": {}
            }
        lst = matrix[input_file][algorithm_name]["all"]
        lst.append({
            "check_result" : check_result,
            "extra_sentinels" : extra_sentinels,
            "mem_local_peak" : algo_peak,
            "mem_local_peak_plus_input_sa": algo_peak + text_peak + sa_peak,
            "duration" : algo_time,
        })

    # Prepare matrix of all gathered data
    algorithms = list(sorted(algorithms))
    threads_and_sizes = list(sorted(threads_and_sizes))
    files = list(sorted(files))

    # Do some post-processing
    for threads_and_size in threads_and_sizes:
        for f in files:
            for algorithm_name in algorithms:
                matrix = outer_matrix[threads_and_size]

                if f not in matrix:
                    matrix[f] = {}
                if algorithm_name not in matrix[f]:
                    matrix[f][algorithm_name] = { "data": "missing" }
                    continue

                data = matrix[f][algorithm_name]
                lst = data["all"]
                avg = data["avg"]
                med = data["med"]

                for key in lst[0]:
                    avg[key] = []
                    med[key] = []

                for e in lst:
                    for key in e:
                        avg[key].append(e[key])
                        med[key].append(e[key])

                def process(t, f):
                    for key in t:
                        if key == "check_result":
                            all_ok = all(e == "ok" for e in t[key])
                            if all_ok:
                                t[key] = "ok"
                            else:
                                t[key] = "error"
                        elif key == "extra_sentinels":
                            assert len(set(t[key])) == 1

                            t[key] = int(t[key][0])
                        else:
                            t[key] = f(t[key])

                process(avg, statistics.mean)
                process(med, statistics.median)

    tex_gen_module.generate_latex_table(outer_matrix, threads_and_sizes, algorithms, files)



# ------------------------------------------------------------------------------

sacabench_exec = "../build/sacabench/sacabench"

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

parser_c = subparsers.add_parser('tablegen', help='tablegen help')
parser_c.add_argument('path', help='path of combined json measure file')
parser_c.add_argument('mode', help='mode')
parser_c.set_defaults(func=handle_tablegen)

args = parser.parse_args()
args.func(args)
