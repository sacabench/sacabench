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
    "2019-04-05T22:32:50",
    #"2019-04-05T22:33:16",
    "2019-04-06T11:30:36",
    "2019-04-06T11:40:44",
    "2019-04-07T15:11:18",
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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        combine_log = ""

        combined_json = []

        gathered = configs[config]["gathered"]
        for key in gathered:
            repetitions = []
            missing_log = []

            # generate missing data annotations for successfull runs as well
            append_all = True

            for datapoint in gathered[key]:
                if not datapoint["stat"] or append_all:
                    err_reason = "no file {}".format("<dummy>")
                    (input, algo, threads) = datapoint["key"]
                    prefix = "<dummy>"

                    outs = ""
                    def log_print(s):
                        nonlocal outs
                        outs += "{}\n".format(s)
                    log_print("Missing data for {}, {}, {}, {} ({})".format(algo, input, prefix, threads, err_reason))
                    log_print("-output----------")
                    log_print(datapoint["output"])
                    log_print("-----------------")

                    missing_log.append(outs)
                if datapoint["stat"]:
                    [single_rep] = datapoint["stat"]
                    repetitions += single_rep

            if len(repetitions) == 0 or append_all:
                combine_log += missing_log[0]

            combined_json.append(repetitions)

            #if len(repetitions) + len(missing_log) != 3:
                #print("{} repetitions: {}, missing_log: {}".format(key, len(repetitions), len(missing_log)))

        config_name = Path(config).stem
        combined_json_path = outdir / Path("{}-results-combined.json".format(config_name))
        combined_log_path = outdir / Path("{}-combine.log".format(config_name))

        write_json(combined_json_path, combined_json)
        write_str(combined_log_path, combine_log)

        #pprint(combined_json)
        #print(combine_log)


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
