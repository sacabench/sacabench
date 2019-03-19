#!/usr/bin/env python3

import subprocess
import argparse
from pathlib import Path
import time
import os
import datetime
import json
from pprint import pprint

dirs = [
    "$HOME/sacabench/external/datasets/downloads",
    "$WORK/large_datasets",
    "../external/datasets/downloads",
]

files = [
    "cc_commoncrawl.ascii.200MB",
    "pc_dblp.xml.200MB",
    "pc_dna.200MB",
    "pc_english.200MB",
    "pc_proteins.200MB",
    "pc_sources.200MB",
    "pcr_cere.200MB",
    "pcr_einstein.en.txt.200MB",
    "pcr_fib41.200MB",
    "pcr_kernel.200MB",
    "pcr_para.200MB",
    "pcr_rs.13.200MB",
    "pcr_tm29.200MB",
    "tagme_wiki-disamb30.200MB",
    "wiki_all_vital.txt.200MB",
    "commoncrawl.txt",
    "dna.txt",
    "wiki.txt",
]

hist_cmd = "../build/sacabench/sacabench histogram -n {path}"

for file in files:
    found = False
    for dir in dirs:
        p = Path(os.path.expandvars(str(Path(dir) / Path(file)))).expanduser()
        if p.is_file():
            found = True
            print("File", p.name)
            subprocess.run(hist_cmd.format(path=p), shell=True, check=True)
            print()
            break;
    if not found:
        print(file, "not found in any directory")
