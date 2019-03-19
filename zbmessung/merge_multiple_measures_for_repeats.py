#!/usr/bin/env python3

import subprocess
import argparse
from pathlib import Path
import time
import os
import datetime
import json
from pprint import pprint

def expand(p):
    return Path(os.path.expandvars(str(Path(p)))).expanduser()

def load_json(path, verbose = False):
    if verbose:
        print("Loading", Path(path))
    with open(path, 'r') as f:
        return json.loads(f.read())

def write_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4))

def write_str(path, data):
    with open(path, 'w') as f:
        f.write(str(data))

def load_str(path, verbose = False):
    if verbose:
        print("Loading", Path(path))
    with open(path, 'r') as f:
        return f.read()

measures = [
    "2019-03-16T13:21:49",
    "2019-03-16T13:24:21",
    "2019-03-16T13:24:28",
    "2019-03-16T18:50:25",
    "2019-03-16T18:50:32",
    "2019-03-16T18:50:42",
    "2019-03-16T21:58:36",
    "2019-03-17T14:53:09",
    "2019-03-17T21:07:14",
    "2019-03-18T00:28:54",
    "2019-03-18T01:00:49",
    "2019-03-18T13:09:12",
]

def handle_extract(args):
    path = args.path
    glob_output = args.output
    # path = "~/uni/lido3/work/smmaloeb/measure"

    configs = {}

    for measure in measures:
        p = expand(Path(path) / Path(measure))
        assert(p.is_dir())

        index_p = p / Path("index.json")
        index = load_json(index_p, verbose=True)

        config = index["launch_config_filename"]
        if not config in configs:
            configs[config] = {
                "raw": [],
                "gathered": {},
            }
        configs[config]["raw"].append(index)
        gathered = configs[config]["gathered"]

        for oof in index["output_files"]:
            threads = int(oof["threads"])
            algo = oof["algo"]
            input = Path(oof["input"]).name

            output = Path(oof["output"]).name
            stat = Path(oof["stat_output"]).name

            full_output = p / Path(output)
            full_stat = p / Path(stat)

            key = (input, algo, threads)

            if not str(key) in gathered:
                gathered[str(key)] = []

            output_data = load_str(full_output, verbose=True)
            stat_data = None
            if full_stat.is_file():
                stat_data = load_json(full_stat, verbose=True)

            gathered[str(key)].append({
                "key": key,
                "stat": stat_data,
                "output": output_data,
            })

    write_json(glob_output, configs)

    return

    for config in configs:
        gathered = configs[config]["gathered"]
        print(config)
        for key in gathered:
            l = len(gathered[key])
            print("{} : {}".format(key, l))

def handle_process(args):
    configs = load_json(args.json, verbose=True)
    outdir = args.outdir

    for config in configs:
        outs = ""
        def log_print(s):
            nonlocal outs
            outs += "{}\n".format(s)

        gathered = configs[config]["gathered"]
        for key in gathered:
            for datapoint in gathered[key]:
                if not datapoint["stat"]:
                    err_reason = "no file {}".format("<dummy>")
                    (input, algo, threads) = datapoint["key"]
                    prefix = "<dummy>"

                    log_print("Missing data for {}, {}, {}, {} ({})".format(algo, input, prefix, threads, err_reason))
                    log_print("-output----------")
                    log_print(datapoint["output"])
                    log_print("-----------------")

        #Missing data for DC3-Parallel-V2, wiki.txt, 200M, 20 (no file stat-inp002-algo001-threads020.json)
        #-output----------
        #Loading input...
        #Running DC3-Parallel-V2 (1/1)
        #terminate called after throwing an instance of 'std::bad_alloc'
        #what():  std::bad_alloc
        #/var/spool/slurm/d/job3543224/slurm_script: line 21: 152922 Aborted                 ./sacabench/sacabench batch /work/smmaloeb/large_datasets/wiki.txt -b /work/smmaloeb/measure/2019-02-24T23:58:32/stat-inp002-algo001-threads020.json -f -s -p 4000M -r 1 --whitelist 'DC3-Parallel-V2' -q -m 64

        #-----------------

        print(outs)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

subparser = subparsers.add_parser('extract')
subparser.add_argument('path')
subparser.add_argument('output')
subparser.set_defaults(func=handle_extract)

subparser = subparsers.add_parser('process')
subparser.add_argument('json')
subparser.add_argument('outdir')
subparser.set_defaults(func=handle_process)

def deflt(args):
    parser.print_help(sys.stderr)
    return 1
parser.set_defaults(func=deflt)

args = parser.parse_args()
args.func(args)
