#!/usr/bin/env python3

import subprocess
import argparse
from pathlib import Path
import time
import os
import datetime

# ---------------------

usage = argparse.ArgumentParser()

usage.add_argument("--launch", action="store_true",
                   help="Wether to actually start task, or merely gather results of previos runs.")
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

args = usage.parse_args()
# ---------------------

batch_template = """#!/bin/bash -l
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
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

    output = os.path.expandvars(str(output))
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    instance = batch_template.format(
        time=args.estimated_time,
        jobname=jobname,
        test_only=test_only,
        cwd=os.path.expandvars(str(cwd)),
        output=output,
        cmd=cmd
    )

    if args.launch:
        if args.print_sbatch:
            print("Instance:\n---\n{}---".format(instance))
        subprocess.run("sbatch", input=instance, encoding="utf-8")

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

if args.launch:
    print("Starting jobs...")

for (j, dataset) in enumerate(datasets):
    for (i, algo) in enumerate(algos):
        sacapath = Path(args.sacabench_directory)

        outdir = Path("$WORK/batch_{}".format(timestamp))
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
        launch_job(cwd, cmd, output)

if args.launch:
    print("Started all jobs!")

print("Current personal job queue:")
subprocess.run("squeue -u $USER", shell=True)
