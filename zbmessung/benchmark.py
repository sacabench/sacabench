#!/usr/bin/env python3

import subprocess
import pprint
import os
import argparse
import json
import statistics
import datetime
import sys

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
            "DSS", # took too long on 1MiB
        ]
    if f == "pcr_rs.13.200MB":
        ret = [
            "Naiv", # took too long on 1MiB
            "Deep-Shallow", # took too long on 1MiB
            "DSS", # took too long on 1MiB
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
            "nzSufSort", # took too long on 1MiB
        ]
    if f == "pcr_einstein.en.txt.200MB":
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
    prefix_size = args.prefix

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

        p = subprocess.Popen(bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in iter(p.stdout.readline, b''):
            sys.stdout.write("{}: {}".format(datetime.datetime.now(), line.decode("utf-8"))),
        p.stdout.close()
        p.wait()

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

                    if f not in matrix:
                        matrix[f] = {}

                    algorithms.add(algorithm_name)
                    if algorithm_name not in matrix[f]:
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


    #pprint.pprint(matrix)
    generate_latex_table(matrix, algorithms, files)

tmp = """
'qsufsort_ref': {'all': [{'check_result': 'ok',
                        'duration': 2856.753294944763,
                        'extra_sentinels': '0',
                        'mem_global_peak': 220200976,
                        'mem_local_peak': 167772176,
                        'mem_local_peak_plus_input_sa': 220200976},
                        {'check_result': 'ok',
                        'duration': 2842.0060551166534,
                        'extra_sentinels': '0',
                        'mem_global_peak': 220200976,
                        'mem_local_peak': 167772176,
                        'mem_local_peak_plus_input_sa': 220200976},
                        {'check_result': 'ok',
                        'duration': 2843.213814020157,
                        'extra_sentinels': '0',
                        'mem_global_peak': 220200976,
                        'mem_local_peak': 167772176,
                        'mem_local_peak_plus_input_sa': 220200976}],
                'avg': {'check_result': 'ok',
                        'duration': 2847.324388027191,
                        'extra_sentinels': 0,
                        'mem_global_peak': 220200976,
                        'mem_local_peak': 167772176,
                        'mem_local_peak_plus_input_sa': 220200976},
                'data': 'exists',
                'med': {'check_result': 'ok',
                        'duration': 2843.213814020157,
                        'extra_sentinels': 0,
                        'mem_global_peak': 220200976,
                        'mem_local_peak': 167772176,
                        'mem_local_peak_plus_input_sa': 220200976}}}}

"""

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

def nice_algoname(n):
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
        #d = d / 60
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
        #d = d / 60
        d = "{:0.2f}".format(d)
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
         """[1:-1], "messung:tab:duration", " in Sekunden"),
        ("Speicherpeak", tex_number("mem_local_peak_plus_input_sa", mem_fmt, "med"), """
\\Cref{%LABEL} entnehmen wir den mediane Speicherverbrauch. Der Wert setzt sich zusammen aus der Allokation für den Eingabetext inklusive der vom Algorithmus benötigten extra Sentinel Bytes, die Allokation für das Ausgabe \sa und dem vom Algorithmus selbst benötigten Speichers.
Pro Eingabe sind erneut die besten drei Algorithmen mit Grün markiert, und die schlechtesten drei mit Rot.
         """[1:-1], "messung:tab:mem", " in MiB"),
    ]

    for (title, get_data, header_text, label, unit) in batch:
        out = generate_latex_table_single(data, algorithms, files, get_data, title, header_text, label, unit)
        print(out)


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
