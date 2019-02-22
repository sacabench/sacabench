#!/usr/bin/env python3

import subprocess
import pprint
import os
import argparse
import json
import statistics
import datetime
import sys

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

    matrix = {}
    algorithms = set()
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
        if input_file not in matrix:
            matrix[input_file] = {}

        algorithms.add(algorithm_name)
        threads_and_sizes.add((thread_count, prefix))

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
    files = list(sorted(matrix))

    pprint.pprint(algorithms)
    pprint.pprint(threads_and_sizes)
    pprint.pprint(files)

    for f in files:
        for algorithm_name in algorithms:
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

            #pprint.pprint([f, algorithm_name])


    pprint.pprint(matrix)
    #generate_latex_table(matrix, algorithms, files)

def latex_rotate(s):
    return "\\rotatebox[origin=c]{{90}}{{{}}}".format(s)

def latex_color(s, cname):
    return "{{\\color{{{}}}{}}}".format(cname, s)

def latex_colorbox(s, cname):
    return "\\colorbox{{{}}}{{{}}}".format(cname, s)

def nice_file(f):
    f = f.replace(".200MB", "")
    f = f.replace("_", "\\_")
    f = "\\texttt{{{}}}".format(f)
    f = latex_rotate(f)
    return f

def fix_algo_name(n):
    if n == "DSS":
        n = "DivSufSort"
    n = n[0].capitalize() + n[1:]
    return n

def nice_algoname(n):
    n = fix_algo_name(n)
    n = "\\text{{{}}}".format(n)
    n = n.replace("_ref", "}_{\\text{ref}}{")
    n = n.replace("{}", "")
    n = "${}$".format(n)
    return n

def generate_latex_table_single(data, algorithms, files, get_data, title, header_text, label, unit):
    out = ""
    out += "\\subsection{{{}}}\n".format(title)
    out += "\n{}\n".format(header_text).replace("%LABEL", label)
    out += "\\begin{table}\n"
    out += "\\caption{{{}{}}}\n".format(title, unit)
    out += "\\label{{{}}}\n".format(label)
    out += "\\resizebox{\\textwidth}{!}{\n"
    out += "\\begin{tabular}{l" + "".join(["r" for e in files]) + "}\n"
    out += "\\toprule\n"

    nice_files = [nice_file(s) for s in files]

    out += "     & {} \\\\\n".format(" & ".join(nice_files))
    out += "\\midrule\n"

    for algorithm_name in algorithms:
        data = list(map(lambda f : get_data(f, algorithm_name), files))
        out += "    {} & {} \\\\\n".format(nice_algoname(algorithm_name), " & ".join(data))

    out += "\\bottomrule\n"
    out += "\\end{tabular}\n"
    out += "}\n"
    out += "\\end{table}\n"

    return out

def generate_latex_table(data, algorithms, files):
    #print("files", len(files))
    #print("algorithms", len(algorithms))

    #sata[f][algorithm_name]["data"]

    def if_check(key, cmp, which):
        nonlocal data

        def ret(f, algorithm_name):
            nonlocal data
            nonlocal cmp
            nonlocal which
            nonlocal key

            if data[f][algorithm_name]["data"] != "exists":
                return "{\color{darkgray}--}"

            if data[f][algorithm_name][which][key] == cmp:
                return "\\cmarkc"
            else:
                return "\\xmarkc"

        return ret

    def tex_number(key, fmt, which):
        nonlocal data
        nonlocal algorithms

        def ret(f, algorithm_name):
            nonlocal data
            nonlocal fmt
            nonlocal which
            nonlocal key
            nonlocal algorithms


            if data[f][algorithm_name]["data"] != "exists":
                return "{\color{darkgray}--}"

            datapoints = set()
            for ai in algorithms:
                if not data[f][ai]["data"] != "exists":
                    d = data[f][ai][which][key]
                    datapoints.add((d, ai))

            datapoints = list(sorted(datapoints, key = lambda t: t[0]))
            datapoints = [(d, n, i) for (i, (d, n)) in enumerate(datapoints)]
            size = len(datapoints)

            (d, i) = next((d, i) for (d, n, i) in datapoints if n == algorithm_name)
            #print((d, i))
            #print()

            raw = data[f][algorithm_name][which][key]
            assert raw == d

            formated = fmt(d, i, size)

            return formated

        return ret

    def time_fmt(d, i, size):
        d = d / 1000
        d = d / 60
        d = "{:0.2f}".format(d)
        #d = latex_rotate("\\ " + d + "\\ ")

        if i < 3:
            d = latex_color(d, "green!60!black")
        elif (size - (i + 1)) < 3:
            d = latex_color(d, "red")

        return d

    def mem_fmt(d, i, size):
        d = d / 1024
        d = d / 1024
        d = d / 1024
        d = "{:0.3f}".format(d)
        #d = latex_rotate("\\ " + d + "\\ ")

        if i < 3:
            d = latex_color(d, "green!60!black")
        elif (size - (i + 1)) < 3:
            d = latex_color(d, "red")

        return d

    batch = [
        #("Measured", if_check("data", "exists", "med")),
        ("\\sa Korrektheit", if_check("check_result", "ok", "med"), """
Wir Überprüfen zunächst, ob alle Testdaten von allen Implementierungen
korrekt verarbeitet werden konnten. Die Messergebnisse enthalten hierfür von \\texttt{-{}-check} erzeugte Informationen und können in \\cref{%LABEL} eingesehen werden.
         """[1:-1], "messung:tab:sa-chk", ""),
        ("Laufzeit", tex_number("duration", time_fmt, "med"), """
Wir betrachten nun in \\cref{%LABEL} die mediane Laufzeit, die jeder Algorithmus für die jeweiligen Testdaten erreicht hat.
Pro Eingabe sind jeweils die besten drei Algorithmen mit Grün markiert, und die schlechtesten drei mit Rot.
         """[1:-1], "messung:tab:duration", " in Minuten"),
        ("Speicherpeak", tex_number("mem_local_peak_plus_input_sa", mem_fmt, "med"), """
\\Cref{%LABEL} entnehmen wir den mediane Speicherverbrauch. Der Wert setzt sich zusammen aus der Allokation für den Eingabetext inklusive der vom Algorithmus benötigten extra Sentinel Bytes, die Allokation für das Ausgabe \sa und dem vom Algorithmus selbst benötigten Speichers.
Pro Eingabe sind erneut die besten drei Algorithmen mit Grün markiert, und die schlechtesten drei mit Rot.
         """[1:-1], "messung:tab:mem", " in GiB"),
    ]

    for (title, get_data, header_text, label, unit) in batch:
        out = generate_latex_table_single(data, algorithms, files, get_data, title, header_text, label, unit)
        print(out)


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
