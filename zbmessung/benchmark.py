#!/usr/bin/env python3

import subprocess
import pprint

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
    if f == "pcr_cere.200MB":
        return ["Deep-Shallow"]

    return []

print("Benchmarking all from:")
pprint.pprint(files)
print("--------------------------------------------")

for f in files:
    full_f = "../external/datasets/downloads/" + f
    full_json = "{}.json".format(f)

    print("Benching {}...".format(full_f))

    blacklist = list(map(lambda x: "--blacklist='{}'".format(x), exceptions(f)))

    bench_cmd = [
        sacabench_exec,
        "batch",

        "--check",
        "--force",
        "--prefix=1M",
        "--benchmark={}".format(full_json),
        *blacklist,

        full_f,
    ]

    plot_cmd = [
        sacabench_exec,
        "plot",
        "batch",
        full_f,
        full_json,
    ]

    subprocess.run(bench_cmd, check=True)
    subprocess.run(plot_cmd, check=True)
    print("--------------------------------------------")
