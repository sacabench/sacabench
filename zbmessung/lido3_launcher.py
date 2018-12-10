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

usage.add_argument("--launch-config", type=Path,
                   help="Config file used by launch.")

cluster_configs = {
    "20cores": {
        "cores": 20,
        "mem": "60G",
        "constraint": "xeon_e52640v4",
    },
    "48cores": {
        "cores": 48,
        "mem": "250G",
        "constraint": "cquad01",
    },
}
cluster_config_default='20cores'
usage.add_argument("--cluster-config", default=cluster_config_default,
                   help="Maximum amount of cores requested. Defaults to {}.".format(cluster_config_default), choices = list(iter(cluster_configs)))

args = usage.parse_args()
# ---------------------

# Note: --parsable means `jobid[;clustername]`
# Also: -Q/--quiet to surpress info messages
batch_template = """#!/bin/bash
#SBATCH --parsable
#SBATCH --quiet
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={maxcores}
#SBATCH --partition=short
#SBATCH --constraint={constraint}
#SBATCH --output={output}
#SBATCH --mem={mem}
#SBATCH --exclusive
#SBATCH --job-name={jobname}
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL
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

    clstcfg = cluster_configs[args.cluster_config]

    instance = batch_template.format(
        time=args.estimated_time,
        jobname=jobname,
        test_only=test_only,
        cwd=cwd,
        output=output,
        cmd=cmd,
        maxcores=clstcfg["cores"],
        mem=clstcfg["mem"],
        constraint=clstcfg["constraint"],
    )

    if args.print_sbatch:
        print("Instance:\n---\n{}---".format(instance))
    jobid = subprocess.check_output("sbatch", input=instance, encoding="utf-8")
    jobid = jobid.strip()
    print("Started job with id {}".format(jobid))
    if jobid != "":
        return jobid
    else:
        return None


timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
WORK = Path(os.path.expandvars("$WORK"))
HOME = Path(os.path.expandvars("$HOME"))
sacapath = Path(os.path.expandvars(args.sacabench_directory))

if args.launch:
    #TODO: Move to external config file
    ALGOS = []
    DATASETS=[]
    N=1
    PREFIX="10M"
    if args.launch_config:
        j = load_json(args.launch_config)
        ALGOS = j["launch"]["algo"]
        DATASETS = j["launch"]["input"]
        N = j["launch"]["rep"]
        PREFIX = j["launch"]["prefix"]

    counter = 0
    print("Starting jobs...")
    index = {
        "output_files" : [],
    }
    outdir = WORK / Path("batch_{}".format(timestamp))
    for (j, dataset) in enumerate(DATASETS):
        for (i, algo) in enumerate(ALGOS):

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
