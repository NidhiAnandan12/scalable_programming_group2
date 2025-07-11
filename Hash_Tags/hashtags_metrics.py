#!/usr/bin/env python3
import argparse
import time
import re
from collections import Counter

from multiprocessing import Pool, cpu_count, set_start_method

import boto3
import matplotlib.pyplot as plt

# pre-compile once
HASHTAG_RE = re.compile(r"#\w+")


def stream_lines(bucket: str, key: str):
    """Yield each line from an S3 object (no local file)."""
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key)
    for raw in resp["Body"].iter_lines(chunk_size=8192):
        yield raw.decode("utf-8", errors="ignore")


def count_hashtags(lines):
    """Count only "meaningful" hashtags (>4 chars or >4 words, containing a letter)."""
    cnt = Counter()
    for line in lines:
        for tag in HASHTAG_RE.findall(line):
            text = tag.lstrip("#")
            parts = text.split("_")
            if (len(text) > 4 or len(parts) > 4) and re.search(r"[A-Za-z]", text):
                cnt[tag] += 1
    return cnt


def sequential_run(lines):
    t0 = time.time()
    cnt = count_hashtags(lines)
    t = time.time() - t0
    return cnt, t


def parallel_worker(idx_range):
    """Unpack a slice of the global LINES list."""
    lo, hi = idx_range
    return count_hashtags(LINES[lo:hi])


def parallel_run(lines, cores):
    """Fork once, then map index-ranges -- avoids large pickles."""
    global LINES
    LINES = lines
    total = len(lines)
    size = total // cores
    ranges = [
        (i*size, (i+1)*size if i < cores-1 else total)
        for i in range(cores)
    ]
    t0 = time.time()
    with Pool(cores) as pool:
        parts = pool.map(parallel_worker, ranges)
    cnt = sum(parts, Counter())
    t = time.time() - t0
    return cnt, t


def plot_bar(cnt, title, out, bucket, key):
    top10 = cnt.most_common(10)
    if not top10:
        print("  (no hashtags)")
        return
    labels, vals = zip(*top10)
    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, vals, edgecolor="black")
    m = max(vals)
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x()+b.get_width()/2,
            h + m*0.01,
            str(int(h)),
            ha="center", va="bottom", fontweight="bold"
        )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Hashtag")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    boto3.client("s3").upload_file(out, bucket, key)
    print(f"Uploaded chart to s3://{bucket}/{key}")


def plot_curve(x, ys, ylabel, title, out, bucket, key, annotations=None):
    """
    x: list of cores
    ys: dict[label]->list of values
    annotations: optional dict[label]->format string
    """
    plt.figure(figsize=(6,4))
    for label, vals in ys.items():
        plt.plot(x, vals, marker='o', label=label)
        if annotations and label in annotations:
            fmt = annotations[label]
            for xi, yi in zip(x, vals):
                plt.text(xi, yi, fmt.format(yi),
                         ha='center', va='bottom', fontweight='bold')
    plt.title(title)
    plt.xlabel("Cores")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    boto3.client("s3").upload_file(out, bucket, key)
    print(f"Uploaded chart to s3://{bucket}/{key}")


if __name__ == "__main__":
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser("Hashtag Performance Metrics")
    parser.add_argument("--filesize", type=float, required=True,
                        help="0.5, 1.0, or 1.5 GB")
    parser.add_argument("--cores", type=str, default="1,2,4,8",
                        help="Comma-separated list of core counts to test")
    args = parser.parse_args()

    bucket = "scalable-project-group2"
    key_map = {
        0.5: "text_files/test_file_05.txt",
        1.0: "text_files/test_file_10.txt",
        1.5: "text_files/test_file_15.txt",
    }
    if args.filesize not in key_map:
        raise SystemExit("Unsupported size")

    s3_key = key_map[args.filesize]
    print("Streaming lines from S3...")
    lines = list(stream_lines(bucket, s3_key))

    # sequential
    print("Running sequential...")
    seq_cnt, t_seq = sequential_run(lines)
    print(f" -> {t_seq:.2f}s")

    # parallel over each core count
    core_list = [int(x) for x in args.cores.split(",")]
    par_times = {}
    for c in core_list:
        print(f"Running parallel on {c} cores...")
        _, t = parallel_run(lines, c)
        par_times[c] = t
        print(f" -> {t:.2f}s")

    # compute metrics
    total = len(lines)
    throughputs = { "Parallel": [ total/par_times[c] for c in core_list ] }
    throughputs["Sequential"] = [ total/t_seq ] * len(core_list)
    latencies  = { "Parallel": [ par_times[c]/total*1e3 for c in core_list ] }
    latencies["Sequential"] = [ t_seq/total*1e3 ] * len(core_list)
    speedups = { "Speedup": [ t_seq/par_times[c] for c in core_list ] }

    # upload top-10 charts
    folder = "hashtag_analysis/hybrid/metrics"
    seq_chart = f"metrics_top10_sequential_{args.filesize}GB.png"
    plot_bar(seq_cnt,
             f"Top-10 Hashtags (Sequential, {args.filesize}GB | {t_seq:.2f}s)",
             seq_chart, bucket, f"{folder}/{seq_chart}")

    max_c = max(core_list)
    par_chart = f"metrics_top10_parallel_{args.filesize}GB_{max_c}cores.png"
    par_cnt, _ = parallel_run(lines, max_c)
    plot_bar(par_cnt,
             f"Top-10 Hashtags (Parallel {max_c} cores, {args.filesize}GB | {par_times[max_c]:.2f}s)",
             par_chart, bucket, f"{folder}/{par_chart}")

    # upload performance curves
    tput_chart = f"metrics_throughput_{args.filesize}GB.png"
    plot_curve(core_list, throughputs,
               "Throughput (lines/sec)",
               f"Throughput vs Cores ({args.filesize}GB)",
               tput_chart, bucket, f"{folder}/{tput_chart}",
               annotations={ "Parallel":"{:.0f}" })

    lat_chart = f"metrics_latency_{args.filesize}GB.png"
    plot_curve(core_list, latencies,
               "Latency (ms/line)",
               f"Latency vs Cores ({args.filesize}GB)",
               lat_chart, bucket, f"{folder}/{lat_chart}",
               annotations={ "Parallel":"{:.3f} ms" })

    speed_chart = f"metrics_speedup_{args.filesize}GB.png"
    plot_curve(core_list, speedups,
               "Speedup (T_seq / T_par)",
               f"Speedup vs Cores ({args.filesize}GB)",
               speed_chart, bucket, f"{folder}/{speed_chart}",
               annotations={ "Speedup":"{:.2f}x" })

    print("All metrics uploaded.")
