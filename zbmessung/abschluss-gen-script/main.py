#!/usr/bin/env python3

import subprocess
from pprint import pprint
import os
import argparse
import json
import statistics
import datetime
import sys
import re

import tex_gen_module

def load_json_from_file(p):
    with open(str(p), 'r') as f:
        json_data = json.load(f)
    return json_data

def save_json_to_file(d, p):
    with open(str(p), 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)

def load_text_from_file(p):
    with open(str(p), 'r') as f:
        data = f.read()
    return data

def save_text_to_file(d, p):
    with open(str(p), 'w') as outfile:
        outfile.write(d)

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

def prepare_and_extract(path):
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

    return {
        "outer_matrix": outer_matrix,
        "threads_and_sizes": threads_and_sizes,
        "algorithms": algorithms,
        "files": files,
    }

def handle_tablegen(args):
    path = args.path
    #mode = args.mode

    processed = prepare_and_extract(path)
    outer_matrix = processed["outer_matrix"]
    threads_and_sizes = processed["threads_and_sizes"]
    algorithms = processed["algorithms"]
    files = processed["files"]

    tex_gen_module.generate_latex_table(outer_matrix, threads_and_sizes, algorithms, files)

def handle_tablegen_all(args):
    cfg = load_json_from_file(args.config)
    pprint(cfg)
    os.makedirs(cfg["output"]["path"], exist_ok=True)

    ttlp = cfg["output"]["table-label-prefix"]
    clp = cfg["output"]["table-check-label-prefix"]
    mlp = cfg["output"]["table-mem-label-prefix"]
    tlp = cfg["output"]["table-time-label-prefix"]

    for measure in cfg["measures"]:
        path = measure["path"]
        mode = measure["thread-scale"]
        processed = prepare_and_extract(path)
        outer_matrix = processed["outer_matrix"]
        threads_and_sizes = processed["threads_and_sizes"]
        algorithms = processed["algorithms"]
        files = processed["files"]

        tls = measure["table-label-suffix"]
        ttitle = measure["title"]

        omit_headings = measure["omit-headings"]

        logfile = measure["log"]
        logdata = parse_logfile(logfile)

        #pprint(logdata)

        table_cfg_kinds = [
            {
                "kind": "sa_check",
                "title": "\\sa Korrektheit {}".format(ttitle),
                "text": "",
                "label": "{}{}{}".format(ttlp, clp, tls),
                "filename": "{}{}".format(clp, tls),
                "omit_headings": omit_headings,
                "logdata": logdata,
                "legend": r"""\begin{tabular}{cl}
\cmarkc & \sa wurde korrekt berechnet.\\
\xmarkc & \sa wurde falsch berechnet.\\
{\color{orange}\faClockO} & Berechnung hat Zeitlimit des Systems erreicht.\\
{\color{purple}\faFloppyO} & Berechnung hat Speicherlimit des Systems erreicht.\\
{\color{violet}\faBolt} & Berechnung brach mit einem Laufzeitfehler ab.\\
\end{tabular}""",
            },
            {
                "kind": "time",
                "title": "Laufzeit in Minuten {}".format(ttitle),
                "text": "",
                "label": "{}{}{}".format(ttlp, tlp, tls),
                "filename": "{}{}".format(tlp, tls),
                "omit_headings": omit_headings,
                "logdata": logdata,
                "legend": r"""\begin{tabular}{ll}
{\color{green}Grün} & Die besten drei Werte.\\
{\color{red}Rot} & Die schlechtesten drei Werte.\\
\end{tabular}""",
            },
            {
                "kind": "mem",
                "title": "Extra-Speicher in GiB {}".format(ttitle),
                "text": "",
                "label": "{}{}{}".format(ttlp, mlp, tls),
                "filename": "{}{}".format(mlp, tls),
                "omit_headings": omit_headings,
                "logdata": logdata,
                "legend": r"""\begin{tabular}{ll}
{\color{green}Grün} & Die besten drei Werte.\\
{\color{red}Rot} & Die schlechtesten drei Werte.\\
\end{tabular}""",
            },
        ]

        for c in table_cfg_kinds:
            e = tex_gen_module.generate_latex_table_list(c, outer_matrix, threads_and_sizes, algorithms, files)
            filename = "{}/{}.tex".format(cfg["output"]["path"], c["filename"])
            save_text_to_file(e, filename)
            print("\\input{{{}}}".format(filename))
            #print(filename)

logfile_re = re.compile("Missing data for (.*?), (.*?), (.*?), (.*?) \\(no file (.*?)\\)\n-output----------\n(.*?)\n-----------------\n", re.DOTALL)
def parse_logfile(logfile):
    text = load_text_from_file(logfile)
    ret = {}
    for m in logfile_re.findall(text):
        algo_name = m[0]
        input_name = m[1]
        input_prefix_str = m[2]
        threads = m[3]
        output = m[5]

        kind = "unknown"

        if "DUE TO TIME LIMIT" in output:
            kind = "timeout"

        # out of memroy patterns
        if "terminate called after throwing an instance of 'std::bad_alloc'" in output:
            kind = "oom"
        if "Some of your processes may have been killed by the cgroup out-of-memory handler." in output:
            kind = "oom"
        if "Allocation Error, not enough space" in output:
            kind = "oom"
        if "terminate called after throwing an instance of 'std::bad_array_new_length'" in output:
            kind = "oom"
        if "malloc failed" in output:
            kind = "oom"

        # crash patterns
        if "Segmentation fault" in output:
            kind = "crash"
        if "double free or corruption" in output:
            kind = "crash"
        if "munmap_chunk(): invalid pointer" in output:
            kind = "crash"
        if "free(): invalid size" in output:
            kind = "crash"

        #print("{}, {}, {}, {}: {}".format(algo_name, input_name, input_prefix_str, threads, kind))
        if kind == "unknown":
            #print(output)
            pass

        assert (algo_name, input_name, threads) not in ret
        ret[(algo_name, input_name, threads)] = kind
    return ret

# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

parser_c = subparsers.add_parser('tablegen-single', help='generate latex tables for single measure file')
parser_c.add_argument('path', help='path of combined json measure file')
parser_c.add_argument('mode', help='mode')
parser_c.set_defaults(func=handle_tablegen)

parser_c = subparsers.add_parser('tablegen-all', help='generate combined')
parser_c.add_argument('--config', help='path of config file', default="config.json")
parser_c.set_defaults(func=handle_tablegen_all)

def deflt(args):
    parser.print_help(sys.stderr)
    return 1
parser.set_defaults(func=deflt)

args = parser.parse_args()
args.func(args)
