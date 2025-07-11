#!/usr/bin/env python3
from multiprocessing import Pool, cpu_count, set_start_method
import argparse
import re
import time
from collections import Counter

import boto3
import matplotlib.pyplot as plt

# Pre-compile once
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

# Worker uses the global LINES list and an index range to avoid pickling the whole list
def count_hashtags_range(idx_range):
    start, end = idx_range
    return count_hashtags(LINES[start:end])

def plot_and_upload(counter, title, out_file, bucket, s3_key):
    top10 = counter.most_common(10)
    if not top10:
        print("No matching hashtags.")
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
    print(f"Uploaded parallel chart to s3://{bucket}/{s3_key}")

if __name__ == "__main__":
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser("Parallel Hashtag Analysis")
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

    key = key_map[args.filesize]
    print("Streaming from S3...")
    lines = list(stream_lines(bucket, key))

    # expose globally for workers
    global LINES
    LINES = lines

    cores = cpu_count()
    print(f"Running parallel on {cores} cores...")
    total = len(LINES)
    chunk_size = total // cores
    ranges = [
        (i*chunk_size, (i+1)*chunk_size if i < cores-1 else total)
        for i in range(cores)
    ]

    t0 = time.time()
    with Pool(processes=cores) as pool:
        parts = pool.map(count_hashtags_range, ranges)
    counter = sum(parts, Counter())
    t_par = time.time() - t0
    print(f"Done in {t_par:.2f}s, found {len(counter)} unique hashtags.")

    title = (
        f"Parallel Hashtag Analysis\n"
        f"Size: {args.filesize} GB | Time: {t_par:.2f} s"
    )
    out_file = f"parallel_hashtags_{args.filesize}GB.png"
    s3_key   = f"hashtag_analysis/parallel/{out_file}"

    plot_and_upload(counter, title, out_file, bucket, s3_key)
