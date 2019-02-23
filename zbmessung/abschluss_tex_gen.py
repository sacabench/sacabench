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

    generate_latex_table(outer_matrix, threads_and_sizes, algorithms, files)

def latex_rotate(s):
    return "\\rotatebox[origin=c]{{90}}{{{}}}".format(s)

def latex_color(s, cname):
    return "{{\\color{{{}}}{}}}".format(cname, s)

def latex_colorbox(s, cname):
    return "\\colorbox{{{}}}{{{}}}".format(cname, s)

def nice_file(f, rotate=True):
    f = f.replace(".200MB", "")
    f = f.replace("_", "\\_")
    f = "\\texttt{{{}}}".format(f)
    if rotate:
        f = latex_rotate(f)
    return f

def nice_algoname(n):
    def fix_algo_name(n):
        if n == "DSS":
            n = "DivSufSort"
        n = n[0].capitalize() + n[1:]
        return n

    n = fix_algo_name(n)
    n = "\\text{{{}}}".format(n)
    #n = n.replace("_par_ref", "}äää{\\text{par,ref}}{")
    n = n.replace("_ref", "}äää{\\text{ref}}{")
    #n = n.replace("_par", "}äää{\\text{par}}{")
    n = n.replace("_", "\\_")
    n = n.replace("äää", "_")
    n = n.replace("{}", "")
    n = "${}$".format(n)
    return n

def expanded_headings_size(headings):
    total_width = 1
    for e in headings:
        total_width *= len(e)
    return total_width

def expand_x_headings(x_headings):
    total_width = expanded_headings_size(x_headings)
    rv = []
    for e in x_headings:
        rv.append([])

    v = rv[0]
    for e in x_headings[0]:
        v.append((e, int(total_width / len(x_headings[0]))))
        if len(x_headings) > 1:
            rrv = expand_x_headings(x_headings[1:])[0]
            for i,sublist in enumerate(rrv):
                rv[i+1] += sublist

    return (rv, total_width)

def generate_latex_table_single_2(multidim_array,
                                  x_headings,
                                  y_headings,
                                  title,
                                  header_text,
                                  label):
    out = ""
    #out += "\\subsection{{{}}}\n".format(title)
    #out += "\n{}\n".format(header_text).replace("%LABEL", label)
    out += "\\begin{table}\n"
    out += "\\resizebox{\\textwidth}{!}{\n"

    assert len(x_headings) >= 1
    assert len(y_headings) == 1

    rv = expand_x_headings(x_headings)
    x_cells = rv[1]
    x_cell_levels = rv[0]

    out += "\\begin{tabular}{l" + ("r" * x_cells) + "}\n"
    out += "\\toprule\n"

    for x_cell_level in x_cell_levels:
        x_cell_level_fmt = []
        for (x_cell, span) in x_cell_level:
            if span == 1:
                x_cell_level_fmt.append(x_cell)
            else:
                x_cell_level_fmt.append("\\multicolumn{{{}}}{{c}}{{{}}}".format(span,x_cell))

        out += "     & {} \\\\\n".format(" & ".join(x_cell_level_fmt))

    out += "\\midrule\n"

    for (y_heading, dataline) in zip(y_headings[0], multidim_array):
        out += "    {} & {} \\\\\n".format(y_heading, " & ".join(dataline))

    out += "\\bottomrule\n"
    out += "\\end{tabular}\n"
    out += "}\n"
    out += "\\caption{{{}}}\n".format(title)
    out += "\\label{{{}}}\n".format(label)
    out += "\\end{table}\n"

    return out

def generate_latex_table(outer_matrix, threads_and_sizes, algorithms, files):
    def if_check(key, cmp, which):
        nonlocal outer_matrix

        def ret(ts, f, algorithm_name):
            nonlocal outer_matrix
            nonlocal cmp
            nonlocal which
            nonlocal key

            if outer_matrix[ts][f][algorithm_name]["data"] != "exists":
                return "{\color{darkgray}--}"

            if outer_matrix[ts][f][algorithm_name][which][key] == cmp:
                return "\\cmarkc"
            else:
                return "\\xmarkc"

        return ret

    def tex_number(key, fmt, which):
        nonlocal outer_matrix
        nonlocal algorithms

        def ret(ts, f, algorithm_name):
            nonlocal outer_matrix
            nonlocal fmt
            nonlocal which
            nonlocal key
            nonlocal algorithms


            if outer_matrix[ts][f][algorithm_name]["data"] != "exists":
                return "{\color{darkgray}--}"

            if outer_matrix[ts][f][algorithm_name][which]["check_result"] != "ok":
                return "{\color{darkgray}--}"

            datapoints = set()
            for ai in algorithms:
                if not outer_matrix[ts][f][ai]["data"] != "exists":
                    if not outer_matrix[ts][f][ai][which]["check_result"] != "ok":
                        d = outer_matrix[ts][f][ai][which][key]
                        datapoints.add((d, ai))

            datapoints = list(sorted(datapoints, key = lambda t: t[0]))
            datapoints = [(d, n, i) for (i, (d, n)) in enumerate(datapoints)]
            size = len(datapoints)

            (d, i) = next((d, i) for (d, n, i) in datapoints if n == algorithm_name)
            #print((d, i))
            #print()

            raw = outer_matrix[ts][f][algorithm_name][which][key]
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
        ("\\sa Korrektheit Weak-Scaling Large", if_check("check_result", "ok", "med"), """
Wir Überprüfen zunächst, ob alle Testdaten von allen Implementierungen
korrekt verarbeitet werden konnten. Die Messergebnisse enthalten hierfür von \\texttt{-{}-check} erzeugte Informationen und können in \\cref{%LABEL} eingesehen werden.
         """[1:-1], "messung:tab:sa-chk-weak-large"),
        ("Laufzeit Weak-Scaling Large in Minuten", tex_number("duration", time_fmt, "med"), """
Wir betrachten nun in \\cref{%LABEL} die mediane Laufzeit, die jeder Algorithmus für die jeweiligen Testdaten erreicht hat.
Pro Eingabe sind jeweils die besten drei Algorithmen mit Grün markiert, und die schlechtesten drei mit Rot.
         """[1:-1], "messung:tab:duration-weak-large"),
        ("Speicherpeak Weak-Scaling Large in GiB", tex_number("mem_local_peak_plus_input_sa", mem_fmt, "med"), """
\\Cref{%LABEL} entnehmen wir den mediane Speicherverbrauch. Der Wert setzt sich zusammen aus der Allokation für den Eingabetext inklusive der vom Algorithmus benötigten extra Sentinel Bytes, die Allokation für das Ausgabe \sa und dem vom Algorithmus selbst benötigten Speichers.
Pro Eingabe sind erneut die besten drei Algorithmen mit Grün markiert, und die schlechtesten drei mit Rot.
         """[1:-1], "messung:tab:mem-weak-large"),
    ]

    for (title, get_data, header_text, label) in batch:
        #out = generate_latex_table_single(data, algorithms, files, get_data, title, header_text, label, unit)

        # data, algorithms, files
        x_tex_headings = [
            [nice_file(s) for s in files],
            ["{}".format(t) for (t, s) in threads_and_sizes],
        ]
        y_tex_headings = [
            [nice_algoname(algorithm_name) for algorithm_name in algorithms],
        ]

        x_headings = [
            files,
            threads_and_sizes,
        ]
        y_headings = [
            algorithms,
        ]

        def rec(headings):
            if len(headings) == 1:
                return [[e] for e in headings[0]]
            else:
                r = []
                for e in headings[0]:
                    ee = rec(headings[1:])
                    for eee in ee:
                        eee.insert(0, e)
                    r += ee
                return r

        multidim_array = []
        for [algorithm] in rec(y_headings):
            multidim_array.append([])
            for [file, threads_and_size] in rec(x_headings):
                #cell = "{}-{}-{}".format(nice_algoname(algorithm), nice_file(file, False), threads_and_size[0])
                cell = get_data(threads_and_size, file, algorithm)
                multidim_array[-1].append(cell)
                #print(algorithm, file, threads_and_size)

        #pprint.pprint(multidim_array)

        out = generate_latex_table_single_2(multidim_array,
                                            x_tex_headings,
                                            y_tex_headings,
                                            title,
                                            header_text,
                                            label)
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
