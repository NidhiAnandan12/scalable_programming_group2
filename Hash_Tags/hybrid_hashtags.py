#!/usr/bin/env python3
import argparse
import time
import re
from collections import Counter
from multiprocessing import Pool, cpu_count, set_start_method

import boto3
import matplotlib.pyplot as plt

# compile once
HASHTAG_RE = re.compile(r"#\w+")

def stream_lines(bucket: str, key: str):
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key)
    for raw in resp["Body"].iter_lines(chunk_size=8192):
        yield raw.decode("utf-8", errors="ignore")

def count_hashtags(lines):
    cnt = Counter()
    for line in lines:
        for tag in HASHTAG_RE.findall(line):
            text = tag.lstrip("#")
            parts = text.split("_")
            if (len(text) > 4 or len(parts) > 4) and re.search(r"[A-Za-z]", text):
                cnt[tag] += 1
    return cnt

# Worker uses global LINES by index range
def count_hashtags_range(idx_range):
    start, end = idx_range
    return count_hashtags(LINES[start:end])

def sequential_hashtag_count(lines):
    return count_hashtags(lines)

def plot_bar(counter, title, out_file, bucket, s3_key):
    top10 = counter.most_common(10)
    if not top10:
        return
    labels, vals = zip(*top10)

    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, vals, edgecolor="black")
    mx = max(vals)
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x()+b.get_width()/2,
            h + mx*0.01,
            str(int(h)),
            ha="center", va="bottom", fontweight="bold"
        )
    plt.title(title)
    plt.xlabel("Hashtag")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

    boto3.client("s3").upload_file(out_file, bucket, s3_key)
    print(f"Uploaded chart to s3://{bucket}/{s3_key}")

def plot_time_comparison(times, title, out_file, bucket, s3_key):
    ordered = [("Parallel", times["parallel"]), ("Sequential", times["sequential"])]
    labels = [l for l, _ in ordered]
    vals   = [v for _, v in ordered]

    plt.figure(figsize=(6,4))
    plt.plot(labels, vals, marker="o")
    for x, y in zip(labels, vals):
        plt.text(x, y, f"{y:.2f}s", ha="center", va="bottom", fontweight="bold")
    plt.title(title)
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

    boto3.client("s3").upload_file(out_file, bucket, s3_key)
    print(f"Uploaded chart to s3://{bucket}/{s3_key}")

if __name__ == "__main__":
    # ensure fork
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser("Hybrid Hashtag Trend Analysis")
    parser.add_argument("--filesize", type=float, required=True,
                        help="0.5, 1.0, or 1.5")
    args = parser.parse_args()

    bucket = "scalable-project-group2"
    key_map = {
        0.5: "text_files/test_file_05.txt",
        1.0: "text_files/test_file_10.txt",
        1.5: "text_files/test_file_15.txt",
    }
    if args.filesize not in key_map:
        print("Unsupported size.")
        exit(1)

    key       = key_map[args.filesize]
    s3_folder = "hashtag_analysis/hybrid"

    print("Streaming from S3...")
    lines = list(stream_lines(bucket, key))
    total = len(lines)
    print(f"Got {total} lines.")

    # expose globally for workers
    global LINES
    LINES = lines

    # Parallel
    cores = cpu_count()
    print(f"Parallel on {cores} cores...")
    # build index ranges, one per core
    chunk_size = total // cores
    ranges = [
        (i*chunk_size, (i+1)*chunk_size if i < cores-1 else total)
        for i in range(cores)
    ]

    t0 = time.time()
    with Pool(cores) as pool:
        parts = pool.map(count_hashtags_range, ranges)
    par_cnt = sum(parts, Counter())
    t_par = time.time() - t0
    print(f"Parallel done in {t_par:.2f}s.")

    par_chart = f"hybrid_parallel_hashtags_{args.filesize}GB.png"
    par_title = (
        f"Parallel Hashtag Analysis\n"
        f"Size: {args.filesize} GB | Time: {t_par:.2f} s"
    )
    plot_bar(par_cnt, par_title, par_chart, bucket, f"{s3_folder}/{par_chart}")

    # Sequential
    print("Sequential...")
    t1 = time.time()
    seq_cnt = sequential_hashtag_count(lines)
    t_seq = time.time() - t1
    print(f"Sequential done in {t_seq:.2f}s.")

    seq_chart = f"hybrid_sequential_hashtags_{args.filesize}GB.png"
    seq_title = (
        f"Sequential Hashtag Analysis\n"
        f"Size: {args.filesize} GB | Time: {t_seq:.2f} s"
    )
    plot_bar(seq_cnt, seq_title, seq_chart, bucket, f"{s3_folder}/{seq_chart}")

    # Comparison
    print("Plotting comparison...")
    cmp_chart = f"hybrid_time_comparison_{args.filesize}GB.png"
    speedup   = (t_seq / t_par) if t_par > 0 else float("inf")
    cmp_title = (
        f"Time Comparison\n"
        f"Size: {args.filesize} GB | Speedup: {speedup:.2f}x"
    )
    plot_time_comparison(
        {"parallel": t_par, "sequential": t_seq},
        cmp_title,
        cmp_chart,
        bucket,
        f"{s3_folder}/{cmp_chart}"
    )
