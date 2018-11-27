#!/usr/bin/env python3

import subprocess
import argparse
from pathlib import Path
import time
import os
import datetime
import json

def load_json(path):
    with open(path, 'r') as f:
        return json.loads(f.read())

def write_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4))

# ---------------------

usage = argparse.ArgumentParser()

usage.add_argument("--launch", action="store_true",
                   help="Launch batch jobs.")
usage.add_argument("--combine", type=Path,
                   help="Gather results of a index.json file produced by a --launch.")
usage.add_argument("--test-only", action="store_true",
                   help="Do not actually start the slurm tasks (do a dry run).")
usage.add_argument("--print-sbatch", action="store_true",
                   help="Print out the batch files used for each job.")

time_default = "02:00:00"
usage.add_argument("--estimated-time", default=time_default,
                   help="Time estimate for the slurm job. Defaults to \"{}\".".format(time_default))

sacabench_default="$HOME/sacabench"
usage.add_argument("--sacabench-directory", default=sacabench_default,
                   help="Location where the sacabench directory is located. Defaults to \"{}\".".format(sacabench_default))

maxcores_default=20  # up to 48
usage.add_argument("--maxcores", default=maxcores_default,
                   type=int,
                   help="Maximum amount of cores requested. Sensible values on lido3 are 20 and 48. Defaults to {}.".format(maxcores_default))

args = usage.parse_args()
# ---------------------

# Note: --parsable means `jobid[;clustername]`
# Also: -Q/--quiet to surpress info messages
batch_template = """#!/bin/bash -l
#SBATCH --parsable
#SBATCH --quiet
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={maxcores}
#SBATCH --partition=short
#SBATCH --constraint=cquad01
#SBATCH --output={output}
#SBATCH --mem=250000
#SBATCH --exclusive
#SBATCH --job-name={jobname}
{test_only}
cd {cwd}
{cmd}
"""

# ---------------------

def launch_job(cwd, cmd, output):
    jobname = "sacabench"

    test_only = ""
    if args.test_only:
        test_only = "#SBATCH --test-only\n"

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    instance = batch_template.format(
        time=args.estimated_time,
        jobname=jobname,
        test_only=test_only,
        cwd=cwd,
        output=output,
        cmd=cmd,
        maxcores=int(args.maxcores)
    )

    if args.print_sbatch:
        print("Instance:\n---\n{}---".format(instance))
    jobid = subprocess.check_output("sbatch", input=instance, encoding="utf-8")
    print("Started job with id {}".format(jobid))
    if jobid != "":
        return jobid
    else:
        return None

#TODO: Move to external config file
algos = [
  "Deep-Shallow_ref",
  "DivSufSort_ref",
  "MSufSort_ref",
  "SACA-K_ref",
  "SADS_ref",
  "SAIS_ref",
  "SAIS-LITE_ref",
  "GSACA_ref",
  "qsufsort_ref",
  "DC3_ref",
  "Deep-Shallow",
  "BPR",
  "BPR_ref",
  "mSufSort",
  "Doubling",
  "Discarding2",
  "Discarding4",
  "Discarding4Parallel",
  "SAIS",
  "SADS",
  "GSACA",
  "GSACA_Opt",
  "GSACA_parallel",
  "DC7",
  "qsufsort",
  "Naiv",
  "SACA-K",
  "DC3",
  "DivSufSort",
  "nzSufSort",
  "DC3-Lite",
  "Osipov_parallel",
  "Osipov_parallel_wp",
]
datasets=[
    #"cc_commoncrawl.ascii.200MB",
    #"combined.txt",
    "pc_dblp.xml.200MB",
    "pc_dna.200MB",
    "pc_english.200MB",
    "pc_proteins.200MB",
    "pc_sources.200MB",
    #"pcr_cere.200MB",
    #"pcr_einstein.en.txt.200MB",
    #"pcr_fib41.200MB",
    #"pcr_kernel.200MB",
    #"pcr_para.200MB",
    #"pcr_rs.13.200MB",
    #"pcr_tm29.200MB",
    #"tagme_wiki-disamb30.200MB",
    #"wiki_all_vital.txt.200MB",
]
N=3
PREFIX="200M"

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
WORK = Path(os.path.expandvars("$WORK"))
HOME = Path(os.path.expandvars("$HOME"))
sacapath = Path(os.path.expandvars(args.sacabench_directory))

if args.launch:
    counter = 0
    print("Starting jobs...")
    index = {
        "output_files" : [],
    }
    outdir = WORK / Path("batch_{}".format(timestamp))
    for (j, dataset) in enumerate(datasets):
        for (i, algo) in enumerate(algos):

            cwd = sacapath / Path("build")
            input_path = sacapath / Path("external/datasets/downloads") / Path(dataset)

            id = "inp{:03}-algo{:03}".format(j, i)

            output = outdir / Path("stdout-{}.txt".format(id))
            batch_output = outdir / Path("stat-{}.json".format(id))

            cmd = "./sacabench/sacabench batch {input_path} -b {bench_out} -f -p {prefix} -r {rep} --whitelist '{algo}'".format(
                bench_out=batch_output,
                prefix=PREFIX,
                rep=N,
                algo=algo,
                input_path=input_path
            )

            jobid = launch_job(cwd, cmd, output)
            index["output_files"].append({
                "output" : str(output),
                "stat_output" : str(batch_output),
                "input": str(input_path),
                "algo": algo,
                "prefix": PREFIX,
                "rep": N,
                "jobid": jobid,
            })

            counter += 1
    write_json(outdir / Path("index.json"), index)
    print("Started {} jobs!".format(counter))
    print("Current personal job queue:")
    subprocess.run("squeue -u $USER", shell=True)

if args.combine:
    dir = Path(args.combine)
    index = load_json(dir / Path("index.json"))
    file_map = {}
    for output_file in index["output_files"]:
        stat_output = Path(output_file["stat_output"])
        algo = output_file["algo"]
        input = Path(output_file["input"])
        prefix = output_file["prefix"]
        if not stat_output.is_file():
            print("Missing data for {}, {}, {} (no file {})".format(algo, input.name, prefix, stat_output.name))
            continue
        stat = load_json(stat_output)
        if not input in file_map:
            file_map[input] = []
        file_map[input] += stat
    for input in file_map:
        write_json(dir / Path("results-{}.json".format(input.name)), file_map[input])
