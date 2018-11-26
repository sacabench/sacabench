#!/usr/bin/env python3

import subprocess
import argparse
from pathlib import Path
import time
import os

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
                   help="Time estimate for the slurm job.  Defaults to \"{}\".".format(time_default))

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

    instance = batch_template.format(
        time=args.estimated_time,
        jobname=jobname,
        test_only=test_only,
        cwd=str(cwd),
        output=str(output),
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

args = usage.parse_args()

if args.launch:
    print("Starting jobs...")
for x in range(0, 10):
    cwd = Path(os.environ["WORK"])
    cmd = "echo test output {}".format(x)
    output = cwd / Path("sacabench_batch_%j.dat")
    launch_job(cwd, cmd, output)
if args.launch:
    print("Started all jobs!")

print("Current personal job queue:")
subprocess.run("squeue -u $USER", shell=True)
