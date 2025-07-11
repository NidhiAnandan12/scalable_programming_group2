#!/usr/bin/env python3
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

def sequential_hashtag_count(lines):
    cnt = Counter()
    for line in lines:
        for tag in HASHTAG_RE.findall(line):
            text = tag.lstrip("#")
            parts = text.split("_")
            # meaningful: more than 4 chars or more than 4 words, must include a letter
            if (len(text) > 4 or len(parts) > 4) and re.search(r"[A-Za-z]", text):
                cnt[tag] += 1
    return cnt

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
    print(f"Uploaded sequential chart to s3://{bucket}/{s3_key}")

def main():
    p = argparse.ArgumentParser("Sequential Hashtag Analysis")
    p.add_argument("--filesize", type=float, required=True,
                   help="0.5, 1.0, or 1.5")
    args = p.parse_args()

    bucket = "scalable-project-group2"
    key_map = {
        0.5: "text_files/test_file_05.txt",
        1.0: "text_files/test_file_10.txt",
        1.5: "text_files/test_file_15.txt",
    }
    if args.filesize not in key_map:
        print("Unsupported size.")
        return

    key = key_map[args.filesize]
    print("Streaming from S3...")
    lines = list(stream_lines(bucket, key))

    print("Running sequential count...")
    t0 = time.time()
    counter = sequential_hashtag_count(lines)
    t_seq = time.time() - t0
    print(f"Done in {t_seq:.2f}s, found {len(counter)} unique hashtags.")

    title = (
        f"Sequential Hashtag Analysis\n"
        f"Size: {args.filesize} GB | Time: {t_seq:.2f} s"
    )
    out_file = f"sequential_hashtags_{args.filesize}GB.png"
    s3_key  = f"hashtag_analysis/sequential/{out_file}"

    plot_and_upload(counter, title, out_file, bucket, s3_key)

if __name__ == "__main__":
    main()
