#!/usr/bin/env python3

import subprocess
import argparse
from pathlib import Path
import time
import os
import datetime
import json
from pprint import pprint

def load_json(path):
    with open(path, 'r') as f:
        return json.loads(f.read())

def write_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4))

def write_str(path, data):
    with open(path, 'w') as f:
        f.write(str(data))

def load_str(path):
    with open(path, 'r') as f:
        return f.read()

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
{extra_args}
{test_only}
{omp_threads}
cd {cwd}
{cmd}
"""

# ---------------------

def launch_job(cwd, cmd, output, omp_threads, clstcfg):
    jobname = "sacabench"

    test_only = ""
    if args.test_only:
        test_only = "#SBATCH --test-only\n"

    omp_threads_str = ""
    if omp_threads:
        omp_threads_str = "export OMP_NUM_THREADS={}\n".format(omp_threads)

    if not args.test_only:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    cores = clstcfg["cores"]
    if omp_threads:
        cores = omp_threads

    instance = batch_template.format(
        time=args.estimated_time,
        jobname=jobname,
        test_only=test_only,
        cwd=cwd,
        output=output,
        cmd=cmd,
        maxcores=cores,
        mem=clstcfg["mem"],
        constraint=clstcfg["constraint"],
        omp_threads=omp_threads_str,
        extra_args="\n".join(map(lambda x: "#SBATCH " + x, clstcfg["extra_args"]))
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
cluster_configs = {
    "20cores": {
        "cores": 20,
        "mem": "60G",
        "constraint": "xeon_e52640v4",
        "extra_args": [],
    },
    "48cores": {
        "cores": 48,
        "mem": "250G",
        "constraint": "xeon_e54640v4",
        "extra_args": [],
    },
    "gpu": {
        "cores": 20,
        "mem": "60G",
        "constraint": "tesla_k40",
        "extra_args": [
            "--gres=gpu:2",
        ],
    },
}
cluster_config_default='20cores'

if args.launch:
    #TODO: Move to external config file
    ALGOS = []
    DATASETS=[]
    N=1
    PREFIX=10
    THREADS=[None]
    WEAK_SCALE = False
    CLUSTER_CONFIG = cluster_config_default
    if args.launch_config:
        j = load_json(args.launch_config)
        ALGOS = j["launch"]["algo"]
        DATASETS = j["launch"]["input"]
        N = j["launch"]["rep"]
        PREFIX = j["launch"]["prefix"]
        if "threads" in j["launch"]:
            THREADS=j["launch"]["threads"]
        if "weak_scale" in j["launch"]:
            WEAK_SCALE = j["launch"]["weak_scale"]
        if "cluster_config" in j["launch"]:
            CLUSTER_CONFIG = j["launch"]["cluster_config"]

    counter = 0
    print("Starting jobs...")
    index = {
        "output_files" : [],
    }
    outdir = WORK / Path("measure/{}".format(timestamp))
    for (j, dataset_path) in enumerate(DATASETS):
        dataset_path = os.path.expandvars(dataset_path)
        dataset_path = Path(dataset_path)
        dataset = dataset_path.name

        for (i, algo) in enumerate(ALGOS):
            for omp_threads in THREADS:
                cwd = sacapath / Path("build")
                input_path = sacapath / Path("external/datasets/downloads") / dataset_path
                input_path = input_path.resolve()

                if omp_threads:
                    threads_str = "threads{:03}".format(omp_threads)
                else:
                    threads_str = "threadsMAX"

                id = "inp{:03}-algo{:03}-{}".format(j, i, threads_str)

                output = outdir / Path("stdout-{}.txt".format(id))
                batch_output = outdir / Path("stat-{}.json".format(id))

                local_prefix = PREFIX
                if omp_threads and WEAK_SCALE:
                    local_prefix *= omp_threads
                local_prefix = "{}M".format(local_prefix)

                cmd = "./sacabench/sacabench batch {input_path} -b {bench_out} -f -p {prefix} -r {rep} --whitelist '{algo}'".format(
                    bench_out=batch_output,
                    prefix=local_prefix,
                    rep=N,
                    algo=algo,
                    input_path=input_path
                )

                jobid = launch_job(cwd, cmd, output, omp_threads, cluster_configs[CLUSTER_CONFIG])
                counter += 1
                index["output_files"].append({
                    "output" : str(output),
                    "stat_output" : str(batch_output),
                    "input": str(input_path),
                    "algo": algo,
                    "prefix": "{}M".format(PREFIX),
                    "actual_prefix": local_prefix,
                    "rep": N,
                    "jobid": jobid,
                    "threads": omp_threads,
                })
    if not args.test_only:
        write_json(outdir / Path("index.json"), index)
        print("Started {} jobs!".format(counter))
        print("Current personal job queue:")
        subprocess.run("squeue -u $USER", shell=True)

def load_data(dir):
    index = load_json(dir / Path("index.json"))
    for output_file in index["output_files"]:
        # Normalize input
        output_file["stat_output"] = Path(output_file["stat_output"])
        output_file["output"] = Path(output_file["output"])
        output_file["input"] = Path(output_file["input"])
        if "threads" not in output_file:
            output_file["threads"] = None

        # Get relevant data
        stat_output = output_file["stat_output"]
        output = output_file["output"]
        algo = output_file["algo"]
        input = output_file["input"]
        prefix = output_file["prefix"]
        threads = output_file["threads"]

        err_reason = ""

        if stat_output.is_file():
            stats = load_json(stat_output)
            if len(stats) == 1:
                yield (output_file, stats[0])
                continue
            else:
                err_reason = "empty json stats"
        else:
            err_reason = "no file {}".format(stat_output.name)

        print("Missing data for {}, {}, {}, {} ({})".format(algo, input.name, prefix, threads, err_reason))
        if output.is_file():
            print("-output----------")
            print(load_str(output))
            print("-----------------")
        continue

def stat_nav_sub(stat, title):
    phase = stat["sub"]
    for e in phase:
        if e["title"] == title:
            return e
    return None

def get_algo_stat(stat):
    #pprint(stat)
    top_stats = extract_stat_logs(stat)
    stat = stat_nav_sub(stat, "SACA")
    saca_stats = extract_stat_logs(stat)
    stat = stat_nav_sub(stat, "Algorithm")
    return {
        "time": stat["timeEnd"] - stat["timeStart"],
        "memPeak": stat["memPeak"],
        "memOff": stat["memOff"],
        "memFinal": stat["memFinal"],
        **top_stats,
        **saca_stats,
    }

def extract_stat_logs(stat):
    l = stat["stats"]
    r = {}
    for e in l:
        r[e["key"]] = e["value"]
    return r

def to_sqlplot(output_file, stats):
    #pprint(stats)
    out = ""
    for (stati, stat) in enumerate(stats):
        o = {
            "algo": output_file["algo"],
            "input": output_file["input"].name,
            "prefix": output_file["prefix"],
            "threads": output_file["threads"],
            "rep": output_file["rep"],
            "rep_i": stati,
            **get_algo_stat(stat),
        }

        s = "RESULT"
        for k in sorted(o):
            s += "\t{}={}".format(k, o[k])
        out += (s + "\n")
    return out

if args.combine:
    dir = Path(args.combine)
    sqlplot_out = ""
    file_map = {}

    combined_json = []
    for (output_file, stats) in load_data(dir):
        threads = output_file["threads"]
        input = output_file["input"]
        sqlplot_out += to_sqlplot(output_file, stats)

        key = (input, str(threads))
        if not key in file_map:
            file_map[key] = []
        file_map[key].append(stats)
    for key in file_map:
        (input, threads) = key
        op = dir / Path("results-{}-{}.json".format(input.name, threads))
        print("Writing data to {}".format(op))
        combined_json.append({
            "threads": threads,
            "input": input.name,
            "stats": file_map[key],
        })
        write_json(op, file_map[key])
    op = dir / Path("sqlplot.txt")
    print("Writing data to {}".format(op))
    write_str(op, sqlplot_out)

    op = dir / Path("results-combined.json")
    print("Writing data to {}".format(op))
    write_json(op, {
        "measures": combined_json,
        "sqlplot": sqlplot_out,
    })
