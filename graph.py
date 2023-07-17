#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import random
import colorsys
import re

file = "./build2/results.csv"
outpath = "./build2/"
title = ""
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='graphs for elementstuffing')
    parser.add_argument('file', type=str, help='result csv')
    parser.add_argument("-o", "--outpath", type=str, help='output path for images', default="./build2/")
    parser.add_argument("-t", "--title", type=str, help='title name specifier for graph', default="")
    args = parser.parse_args()
    file = args.file
    outpath = args.outpath + "/"
    title = args.title + "\n"

data = pd.read_csv(file, delimiter=";")
data = data.groupby(["approach", "element_count", "bits", "block_count", "thread_count"]).mean().reset_index()

def gen_plots(path_suffix, data_size, data_approaches, data_approach_list, data_bits, runtime_no_throughput):
    # scatter plot of all runs and configs
    sp1, spa1 = plt.subplots(figsize=(16,12))
    bits_offset = 0
    for approach in data_approach_list:
        spa1.scatter(data_approaches[approach].bits + bits_offset, data_approaches[approach].time_ms if runtime_no_throughput else data_approaches[approach].throughput, label=approach, s=3)
        bits_offset += 0.1
    spa1.legend()
    spa1.set_xlabel("bits")
    spa1.set_xticks(data_bits)
    spa1.set_ylabel("runtime (ms)" if runtime_no_throughput else "throughput (GiB/s)")
    spa1.set_title(f"{title}elementstuffing{path_suffix} approaches, {'runtime' if runtime_no_throughput else 'throughput'} over bits ({data_size / 1024 / 1024} MiB)")
    sp1.savefig(f"{outpath}scatter{path_suffix}.png")

    # bar chart over best in class configurations
    # sp2, spa2 = plt.subplots(figsize=(16,12))
    # bic_bits_acc = dict([(b, {}) for b in data_bits])
    # approach_labels = {}
    # random.seed(17)
    approach_colors = dict([(a, colorsys.hsv_to_rgb(1 / len(data_approach_list) * i,1,1)) for i, a in enumerate(data_approach_list)])
    # approach_labeled = dict([(a, False) for a in data_approach_list])
    # for approach in data_approach_list:
    #     bic_bits = data_approaches[approach].groupby(["bits"]).max().reset_index()
    #     bic_max_throughput_config = data_approaches[approach].groupby(["block_count", "thread_count"]).mean()["throughput"].idxmax()
    #     for bit in bic_bits.bits:
    #         bic_bits_acc[bit][approach] = bic_bits[(bic_bits.bits == bit) & (bic_bits.approach == approach)].reset_index().throughput[0]
    #     approach_labels[approach] = f"{approach} {bic_max_throughput_config}"
    # for bit, entry in bic_bits_acc.items():
    #     throughputlist = reversed(sorted((throughput, approach) for (approach, throughput) in entry.items()))
    #     for throughput, approach in throughputlist:
    #         spa2.bar(bit, throughput, color=approach_colors[approach], label=approach_labels[approach] if not approach_labeled[approach] else None)
    #         approach_labeled[approach] = True
    # spa2.legend()
    # spa2.set_xlabel("bits")
    # spa2.set_xticks(data_bits)
    # spa2.set_ylabel("throughput (GiB/s)")
    # spa2.set_title(f"{title}elementstuffing{path_suffix} approaches best-mean-in-class configuration, throughput over bits ({data_size / 1024 / 1024} MiB)")
    # sp2.savefig(f"{outpath}bars{path_suffix}.png")

    # line graph over best in class configurations
    sp3, spa3 = plt.subplots(figsize=(16,12))
    for approach in data_approach_list:
        bic_bits = data_approaches[approach].groupby(["bits"])
        if runtime_no_throughput:
            bic_bits = bic_bits.min()
        else:
            bic_bits = bic_bits.max()
        bic_bits = bic_bits.reset_index()
        bic_max_throughput_config = data_approaches[approach].groupby(["block_count", "thread_count"]).mean()["throughput"].idxmax()
        spa3.plot(bic_bits.bits, bic_bits.time_ms if runtime_no_throughput else bic_bits.throughput, color=approach_colors[approach], label=f"{approach} {bic_max_throughput_config}")
    spa3.legend()
    spa3.set_xlabel("bits")
    spa3.set_xticks(data_bits)
    spa3.set_ylabel("runtime (ms)" if runtime_no_throughput else "throughput (GiB/s)")
    spa3.set_title(f"{title} elementstuffing{path_suffix} approaches best-mean-in-class configuration, {'runtime' if runtime_no_throughput else 'throughput'} over bits ({data_size / 1024 / 1024} MiB)")
    sp3.savefig(f"{outpath}lines{path_suffix}.png")


operations = ["binary_op", "compressstore", "filter", "groupby", "hop"]
data_size = int(data.element_count.unique()[0]) * 8  # sizeof(uint64_t)
approaches = data.approach.unique()
data_bits = data.bits.unique()
for op in operations:
    op_approaches = [ap for ap in approaches if op in ap]
    data_by_approach = {}
    for ap in op_approaches:
        data_by_approach[ap] = data[data.approach == ap]
    if max((len(dt) for (_, dt) in data_by_approach.items()), default=0) == 0:
        continue
    ds = data_size if op != "binary_op" else 2 * data_size
    gen_plots(op, ds, data_by_approach, op_approaches, data_bits, True)
    gen_plots(op + "_tp", ds, data_by_approach, op_approaches, data_bits, False)
