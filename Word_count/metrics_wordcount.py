#!/usr/bin/env python3
import argparse
import time
from collections import Counter
from multiprocessing import Pool, cpu_count, set_start_method

import boto3
import matplotlib.pyplot as plt


def stream_lines(bucket: str, key: str):
    """Yield each line from an S3 object without saving locally."""
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key)
    for raw in resp["Body"].iter_lines(chunk_size=8192):
        yield raw.decode("utf-8", errors="ignore")


def count_words(lines):
    """Count words in a list of lines."""
    cnt = Counter()
    for line in lines:
        cnt.update(line.strip().lower().split())
    return cnt


def sequential_run(lines):
    """Sequential word count with timing."""
    t0 = time.time()
    cnt = count_words(lines)
    t = time.time() - t0
    return cnt, t


def parallel_worker(idx_range):
    """Worker that processes a slice of the global LINES list."""
    lo, hi = idx_range
    return count_words(LINES[lo:hi])


def parallel_run(lines, cores):
    """Fork once, then map index-ranges -- avoids large list pickles."""
    global LINES
    LINES = lines
    total = len(lines)
    size = total // cores
    ranges = [
        (i * size, (i + 1) * size if i < cores - 1 else total)
        for i in range(cores)
    ]
    t0 = time.time()
    with Pool(cores) as pool:
        parts = pool.map(parallel_worker, ranges)
    cnt = sum(parts, Counter())
    t = time.time() - t0
    return cnt, t


def plot_bar(cnt, title, out_file, bucket, s3_key):
    """Plot top-10 counts as bar chart and upload to S3."""
    top10 = cnt.most_common(10)
    if not top10:
        print("  (no words)")
        return
    words, vals = zip(*top10)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(words, vals, edgecolor="black")
    mx = max(vals)
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            h + mx * 0.01,
            str(int(h)),
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    plt.title(title)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    boto3.client("s3").upload_file(out_file, bucket, s3_key)
    print(f"Uploaded chart to s3://{bucket}/{s3_key}")


def plot_curve(x, series, ylabel, title, out_file, bucket, s3_key, fmt=None):
    """
    Plot one or more series vs x.
    series: dict[label] -> list of values
    fmt: dict[label] -> annotation format string
    """
    plt.figure(figsize=(6, 4))
    for label, vals in series.items():
        plt.plot(x, vals, marker="o", label=label)
        if fmt and label in fmt:
            for xi, yi in zip(x, vals):
                plt.text(xi, yi, fmt[label].format(yi),
                         ha="center", va="bottom", fontweight="bold")
    plt.title(title)
    plt.xlabel("Cores")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    boto3.client("s3").upload_file(out_file, bucket, s3_key)
    print(f"Uploaded chart to s3://{bucket}/{s3_key}")


if __name__ == "__main__":
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser("Word Count Performance Metrics")
    parser.add_argument(
        "--filesize", type=float, required=True,
        help="0.5, 1.0, or 1.5 GB"
    )
    parser.add_argument(
        "--cores", type=str, default="1,2,4,8",
        help="Comma-separated list of core counts"
    )
    args = parser.parse_args()

    bucket = "scalable-project-group2"
    key_map = {
        0.5: "text_files/test_file_05.txt",
        1.0: "text_files/test_file_10.txt",
        1.5: "text_files/test_file_15.txt",
    }
    if args.filesize not in key_map:
        raise SystemExit("Unsupported file size")

    input_key = key_map[args.filesize]
    folder = f"word_count_graphs/hybrid/metrics"

    # 1) Load lines once
    print("Streaming lines from S3...")
    lines = list(stream_lines(bucket, input_key))
    total = len(lines)
    print(f"Retrieved {total} lines.")

    # 2) Sequential run
    print("Running sequential word count...")
    seq_cnt, t_seq = sequential_run(lines)
    print(f" -> {t_seq:.2f}s")

    # 3) Parallel runs
    core_list = [int(x) for x in args.cores.split(",")]
    par_times = {}
    for c in core_list:
        print(f"Running parallel on {c} cores...")
        _, t = parallel_run(lines, c)
        par_times[c] = t
        print(f" -> {t:.2f}s")

    # 4) Compute metrics
    throughput = {
        "Parallel": [total / par_times[c] for c in core_list],
        "Sequential": [total / t_seq] * len(core_list)
    }
    latency = {
        "Parallel": [par_times[c] / total * 1e3 for c in core_list],
        "Sequential": [t_seq / total * 1e3] * len(core_list)
    }
    speedup = {
        "Speedup": [t_seq / par_times[c] for c in core_list]
    }

    # 5) Top-10 bar charts
    seq_chart = f"metrics_top10_sequential_{args.filesize}GB.png"
    plot_bar(
        seq_cnt,
        f"Top-10 Words (Sequential, {args.filesize}GB | {t_seq:.2f}s)",
        seq_chart, bucket, f"{folder}/{seq_chart}"
    )

    max_c = max(core_list)
    par_cnt, _ = parallel_run(lines, max_c)
    par_chart = f"metrics_top10_parallel_{args.filesize}GB_{max_c}cores.png"
    plot_bar(
        par_cnt,
        f"Top-10 Words (Parallel {max_c} cores, {args.filesize}GB | {par_times[max_c]:.2f}s)",
        par_chart, bucket, f"{folder}/{par_chart}"
    )

    # 6) Performance curves
    tput_chart = f"metrics_throughput_{args.filesize}GB.png"
    plot_curve(
        core_list, throughput,
        "Throughput (lines/sec)",
        f"Throughput vs Cores ({args.filesize}GB)",
        tput_chart, bucket, f"{folder}/{tput_chart}",
        fmt={"Parallel":"{:.0f}"}
    )

    lat_chart = f"metrics_latency_{args.filesize}GB.png"
    plot_curve(
        core_list, latency,
        "Latency (ms/line)",
        f"Latency vs Cores ({args.filesize}GB)",
        lat_chart, bucket, f"{folder}/{lat_chart}",
        fmt={"Parallel":"{:.3f} ms"}
    )

    spd_chart = f"metrics_speedup_{args.filesize}GB.png"
    plot_curve(
        core_list, speedup,
        "Speedup (T_seq / T_par)",
        f"Speedup vs Cores ({args.filesize}GB)",
        spd_chart, bucket, f"{folder}/{spd_chart}",
        fmt={"Speedup":"{:.2f}x"}
    )

    print("All metrics uploaded.")
