#!/usr/bin/env python3
import argparse
import time
import re
from collections import Counter
from multiprocessing import Pool, cpu_count, set_start_method

import boto3
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# ensure the VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def stream_lines(bucket: str, key: str):
    """Yield each line from an S3 object as UTF-8 text."""
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key)
    for raw in resp["Body"].iter_lines(chunk_size=8192):
        yield raw.decode("utf-8", errors="ignore")

def init_worker():
    """Initialize SentimentIntensityAnalyzer in each parallel worker."""
    global analyzer
    analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(lines):
    """
    Count positive/negative/neutral lines in `lines`.
    Positive: compound>=0.05; Negative: <=-0.05; else neutral.
    """
    cnt = Counter()
    for line in lines:
        comp = analyzer.polarity_scores(line)["compound"]
        if comp >= 0.05:
            cnt["positive"] += 1
        elif comp <= -0.05:
            cnt["negative"] += 1
        else:
            cnt["neutral"] += 1
    return cnt

def sequential_run(lines):
    """Run sentiment classification sequentially and measure time."""
    t0 = time.time()
    cnt = classify_sentiment(lines)
    t = time.time() - t0
    return cnt, t

def parallel_worker(idx_range):
    """Parallel worker: classify a slice of the global LINES list."""
    lo, hi = idx_range
    return classify_sentiment(LINES[lo:hi])

def parallel_run(lines, cores):
    """
    Run classification in parallel over `cores` workers.
    Uses global LINES to avoid pickling the entire list.
    """
    global LINES
    LINES = lines
    total = len(lines)
    size = total // cores
    ranges = [
        (i*size, (i+1)*size if i < cores-1 else total)
        for i in range(cores)
    ]
    t0 = time.time()
    with Pool(cores, initializer=init_worker) as pool:
        parts = pool.map(parallel_worker, ranges)
    cnt = sum(parts, Counter())
    t = time.time() - t0
    return cnt, t

def plot_bar(cnt, title, out_file, bucket, s3_key):
    """Plot a 3-bar chart (positive/negative/neutral) and upload to S3."""
    labels = ["positive","negative","neutral"]
    vals = [cnt.get(l,0) for l in labels]
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, vals, edgecolor="black")
    mx = max(vals) if vals else 1
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x()+b.get_width()/2,
            h + mx*0.01,
            str(int(h)),
            ha="center", va="bottom", fontweight="bold"
        )
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    boto3.client("s3").upload_file(out_file, bucket, s3_key)
    print(f"Uploaded chart to s3://{bucket}/{s3_key}")

def plot_curve(x, series, ylabel, title, out_file, bucket, s3_key, fmt=None):
    """
    Plot multiple series vs x and upload to S3.
    fmt: optional dict[label]->format string for annotations
    """
    plt.figure(figsize=(6,4))
    for label, vals in series.items():
        plt.plot(x, vals, marker='o', label=label)
        if fmt and label in fmt:
            for xi, yi in zip(x, vals):
                plt.text(xi, yi, fmt[label].format(yi),
                         ha='center', va='bottom', fontweight='bold')
    plt.title(title)
    plt.xlabel("Cores")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    boto3.client("s3").upload_file(out_file, bucket, s3_key)
    print(f"Uploaded chart to s3://{bucket}/{s3_key}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser("Sentiment Performance Metrics")
    parser.add_argument("--filesize", type=float, required=True,
                        help="0.5, 1.0, or 1.5 GB")
    parser.add_argument("--cores", type=str, default="1,2,4,8",
                        help="Comma-separated list of core counts")
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
    folder = "sentiment_analysis/hybrid/metrics"

    # 1) Stream the lines from S3
    print("Streaming lines from S3...")
    lines = list(stream_lines(bucket, input_key))
    total = len(lines)
    print(f"Retrieved {total} lines.")

    # 2) Initialize analyzer for the sequential path
    analyzer = SentimentIntensityAnalyzer()

    # 3) Sequential run
    print("Running sequential sentiment analysis...")
    seq_cnt, t_seq = sequential_run(lines)
    print(f" -> {t_seq:.2f}s")

    # 4) Parallel runs for each specified core count
    core_list = [int(x) for x in args.cores.split(",")]
    par_times = {}
    for c in core_list:
        print(f"Running parallel on {c} cores...")
        _, t = parallel_run(lines, c)
        par_times[c] = t
        print(f" -> {t:.2f}s")

    # 5) Compute throughput, latency, speedup
    throughput = {
        "Parallel": [ total/par_times[c] for c in core_list ],
        "Sequential": [ total/t_seq ] * len(core_list)
    }
    latency = {
        "Parallel": [ par_times[c]/total*1e3 for c in core_list ],
        "Sequential":[ t_seq/total*1e3 ] * len(core_list)
    }
    speedup = { "Speedup": [ t_seq/par_times[c] for c in core_list ] }

    # 6) Upload bar charts
    seq_chart = f"metrics_bar_sequential_{args.filesize}GB.png"
    plot_bar(
        seq_cnt,
        f"Sentiment (Sequential, {args.filesize}GB | {t_seq:.2f}s)",
        seq_chart, bucket, f"{folder}/{seq_chart}"
    )

    max_c = max(core_list)
    par_cnt, _ = parallel_run(lines, max_c)
    par_chart = f"metrics_bar_parallel_{args.filesize}GB_{max_c}cores.png"
    plot_bar(
        par_cnt,
        f"Sentiment (Parallel {max_c} cores, {args.filesize}GB | {par_times[max_c]:.2f}s)",
        par_chart, bucket, f"{folder}/{par_chart}"
    )

    # 7) Upload performance curves
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

    print("All sentiment metrics uploaded.")
