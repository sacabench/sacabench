#!/usr/bin/env python3

import subprocess
import pprint
import os

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

sacabench_exec = "../build/sacabench/sacabench"

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

print("Benchmarking all from:")
pprint.pprint(files)
print("Blacklists:")
blacklists = []
for f in files:
    tmp = exceptions(f)
    if len(tmp) != 0:
        blacklists += [(f, exceptions(f))]
pprint.pprint(blacklists)

prefix_size = "10M"
repetitions = "3"
make_plot = False

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

    plot_cmd = [
        sacabench_exec,
        "plot",
        "batch",
        full_f,
        full_json,
        "--prefix", prefix_size,
    ]

    print("Run {}".format(bench_cmd))
    subprocess.run(bench_cmd, check=False)

    if make_plot:
        print("Run {}".format(plot_cmd))
        subprocess.run(plot_cmd, check=True)
        subprocess.run(["mv", full_json + " .pdf", full_json + ".pdf"], check=True)

    print("--------------------------------------------")
