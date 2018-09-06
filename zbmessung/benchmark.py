#!/usr/bin/env python3

import subprocess
import pprint
import os
import argparse
import json
import statistics

# options
# - sa_bits: 32, 40, 64
# - prefix size: 200, larger?
# - repetitions
# - blacklist?

files = [
    "cc_commoncrawl.ascii.200MB",
    "pc_dblp.xml.200MB",
    "pc_dna.200MB",
    "pc_english.200MB",
    "pc_proteins.200MB",
    "pcr_cere.200MB",
    "pcr_einstein.en.txt.200MB",
    "pcr_fib41.200MB",
    "pcr_kernel.200MB",
    "pcr_para.200MB",
    "pcr_rs.13.200MB",
    "pcr_tm29.200MB",
    "pc_sources.200MB",
    "tagme_wiki-disamb30.200MB",
    "wiki_all_vital.txt.200MB",
]

def exceptions(f):
    ret = []

    if f == "pcr_cere.200MB":
        ret = [
            "Deep-Shallow", # took too long on 1MiB
            "mSufSort", # took too long on 1MiB
            "nzSufSort", # took too long on 1MiB
        ]
    if f == "pcr_fib41.200MB":
        ret = [
            "Naiv", # took too long on 1MiB
            "Deep-Shallow", # took too long on 1MiB
        ]
    if f == "pcr_rs.13.200MB":
        ret = [
            "Naiv", # took too long on 1MiB
            "Deep-Shallow", # took too long on 1MiB
        ]
    if f == "pcr_tm29.200MB":
        ret = [
            "Naiv", # took too long on 1MiB
            "DSS", # took too long on 1MiB
            "Deep-Shallow", # took too long on 1MiB
        ]
    if f == "pcr_para.200MB":
        ret = [
            "Deep-Shallow", # took too long on 1MiB
        ]

    ret += ["Doubling"] # redundant, and takes too long in general

    return ret

def load_json_from_file(p):
    with open(str(p), 'r') as f:
        json_data = json.load(f)
    return json_data

def save_json_to_file(d, p):
    with open(str(p), 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)

def handle_measure(args):
    repetitions = args.repetitions
    prefix_size = args.prefix_size

    print("Benchmarking all from:")
    pprint.pprint(files)
    print("Blacklists:")
    blacklists = []
    for f in files:
        tmp = exceptions(f)
        if len(tmp) != 0:
            blacklists += [(f, exceptions(f))]
    pprint.pprint(blacklists)

    print("Prefix size: {}".format(prefix_size))
    print("Repetitions: {}".format(repetitions))

    print("--------------------------------------------")

    for f in files:
        measures_dir = "measures/size-{}-rep-{}".format(prefix_size, repetitions)
        full_f = "../external/datasets/downloads/" + f
        full_json = "{}/{}.json".format(measures_dir, f)
        blacklist = exceptions(f)

        subprocess.run(["mkdir", "-p", measures_dir], check=True)

        hsh = [prefix_size, repetitions, full_f, full_json, blacklist]
        print("Hash: {}".format(hsh))

        print("Benching {}...".format(full_f))
        print("Blacklisting the following algorithms:")
        pprint.pprint(blacklist)

        if os.path.isfile(full_json):
            print("Already done")
            continue

        blacklist_args = []
        for blacklist_arg in map(lambda x: ["--blacklist", x], blacklist):
            blacklist_args += blacklist_arg

        bench_cmd = [
            sacabench_exec,
            "batch",

            "--check",
            "--force",
            "--prefix", prefix_size,
            "--repetitions", repetitions,
            "--benchmark", full_json,
            *blacklist_args,

            full_f,
        ]

        print("Run {}".format(bench_cmd))
        subprocess.run(bench_cmd, check=False)

        print("--------------------------------------------")

def handle_plot(args):
    path = args.path
    ps = args.prefix
    for f in files:
        measures_dir = path
        full_f = "../external/datasets/downloads/" + f
        full_json = "{}/{}.json".format(measures_dir, f)
        if os.path.isfile(full_json):
            plot_cmd = [
                sacabench_exec,
                "plot",
                "batch",
                full_f,
                full_json,
                "--prefix", ps,
            ]
            print("Run {}".format(plot_cmd))
            subprocess.run(plot_cmd, check=True)
            subprocess.run(["mv", full_json + " .pdf", full_json + ".pdf"], check=True)

def get_stats(stats):
    mp = {}
    for s in stats:
        if not s["key"] in mp:
            mp[s["key"]] = []
        mp[s["key"]].append(s["value"])
    return mp

def handle_tablegen(args):
    path = args.path
    ps = args.prefix

    matrix = {}

    algorithms = set()

    for f in files:
        measures_dir = path
        full_f = "../external/datasets/downloads/" + f
        full_json = "{}/{}.json".format(measures_dir, f)
        if os.path.isfile(full_json):
            js = load_json_from_file(full_json)
            for algos in js:
                for algo in algos:
                    root_stats = get_stats(algo["stats"])
                    algorithm_name = root_stats["algorithm_name"][0]

                    saca_phase = algo["sub"][0]
                    assert(saca_phase["title"] == "SACA")

                    checker_phase = algo["sub"][1]
                    assert(checker_phase["title"] == "SA Checker")
                    check_result = get_stats(checker_phase["stats"])["check_result"][0]

                    saca_phase_stats = get_stats(saca_phase["stats"])
                    saca_phase_stats_extra_sentinels = saca_phase_stats["extra_sentinels"][0]

                    algorithm_phase =  saca_phase["sub"][3]
                    assert(algorithm_phase["title"] == "Algorithm")

                    alloc_phase =  saca_phase["sub"][0]
                    assert(alloc_phase["title"] == "Allocate SA and Text container")
                    alloc_phase_peak = alloc_phase["memPeak"]

                    algorithm_phase_mem_final = algorithm_phase["memFinal"]
                    algorithm_phase_mem_peak = algorithm_phase["memPeak"]
                    algorithm_phase_mem_off = algorithm_phase["memOff"]

                    algorithm_phase_time_start = algorithm_phase["timeStart"]
                    algorithm_phase_time_end = algorithm_phase["timeEnd"]
                    algorithm_phase_time_duration = algorithm_phase_time_end - algorithm_phase_time_start

                    if not f in matrix:
                        matrix[f] = {}

                    algorithms.add(algorithm_name)
                    if not algorithm_name in matrix[f]:
                        matrix[f][algorithm_name] = { "data" : "exists", "all" : [], "avg": {}, "med": {} }

                    lst = matrix[f][algorithm_name]["all"]

                    lst.append({
                        "check_result" : check_result,
                        "extra_sentinels" : saca_phase_stats_extra_sentinels,
                        #"mem_final" : algorithm_phase_mem_final,
                        "mem_local_peak" : algorithm_phase_mem_peak,
                        "mem_local_peak_plus_input_sa": alloc_phase_peak + algorithm_phase_mem_peak,
                        #"mem_off" : algorithm_phase_mem_off,
                        "mem_global_peak" : algorithm_phase_mem_off + algorithm_phase_mem_peak,
                        "duration" : algorithm_phase_time_duration,
                    })

    algorithms = list(sorted(algorithms))

    for f in matrix:
        for algorithm_name in algorithms:
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

            #pprint.pprint([f, algorithm_name, lst])

    pprint.pprint(matrix)

# ------------------------------------------------------------------------------

sacabench_exec = "../build/sacabench/sacabench"

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

parser_a = subparsers.add_parser('measure', help='measure help')
parser_a.add_argument('prefix', help='prefix size of data')
parser_a.add_argument('repetitions', help='repetitions of measures')
parser_a.set_defaults(func=handle_measure)

parser_b = subparsers.add_parser('plot', help='plot help')
parser_b.add_argument('path', help='path of measure directory')
parser_b.add_argument('prefix', help='prefix size of data')
parser_b.set_defaults(func=handle_plot)

parser_c = subparsers.add_parser('tablegen', help='tablegen help')
parser_c.add_argument('path', help='path of measure directory')
parser_c.add_argument('prefix', help='prefix size of data')
parser_c.set_defaults(func=handle_tablegen)

args = parser.parse_args()
args.func(args)
