import argparse
import time
from collections import Counter
from multiprocessing import Pool, cpu_count, set_start_method

import boto3
import matplotlib.pyplot as plt


def stream_lines_from_s3(bucket: str, key: str):
    """
    Yield each line of an S3 object as a decoded UTF-8 string,
    without saving the whole file locally.
    """
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key)
    for raw in resp["Body"].iter_lines(chunk_size=8192):
        yield raw.decode("utf-8", errors="ignore")


def analyze_sentiment(line: str) -> float:
    """
    Return the polarity score of a single line using TextBlob.
    """
    from textblob import TextBlob
    return TextBlob(line).sentiment.polarity


def classify_counts(sentiments):
    """
    Given a list of polarity floats, return a Counter of
    'positive', 'negative', 'neutral'.
    """
    cnt = Counter()
    for s in sentiments:
        if s > 0:
            cnt['positive'] += 1
        elif s < 0:
            cnt['negative'] += 1
        else:
            cnt['neutral'] += 1
    return cnt


def plot_bar(counter: Counter, title: str, out_path: str):
    """
    Plot a bar chart for the three sentiment counts.
    """
    labels = ['positive', 'negative', 'neutral']
    values = [counter.get(l, 0) for l in labels]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, edgecolor='black')
    max_val = max(values) if values else 1
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_val * 0.01,
            str(int(h)),
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_time_comparison(times: dict, title: str, out_path: str):
    """
    Plot a line chart comparing two times.
    """
    ordered = [("Parallel", times['parallel']), ("Sequential", times['sequential'])]
    labels = [o[0] for o in ordered]
    secs   = [o[1] for o in ordered]

    plt.figure(figsize=(6, 4))
    plt.plot(labels, secs, marker='o')
    for x, y in zip(labels, secs):
        plt.text(x, y, f"{y:.2f}s", ha='center', va='bottom', fontweight='bold')

    plt.title(title)
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    # ensure fork start method on Linux
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Hybrid Sentiment Analysis from S3")
    parser.add_argument(
        "--filesize",
        type=float,
        required=True,
        help="File size identifier: 0.5, 1.0, 1.5, etc."
    )
    args = parser.parse_args()

    bucket = "scalable-project-group2"
    size = args.filesize
    key_map = {
        0.5: "text_files/test_file_05.txt",
        1.0: "text_files/test_file_10.txt",
        1.5: "text_files/test_file_15.txt",
    }
    if size not in key_map:
        print("Unsupported file size.")
        exit(1)

    input_key = key_map[size]
    s3_folder = "sentiment_analysis/hybrid"

    # stream lines from S3
    print("Fetching lines from S3...")
    lines = list(stream_lines_from_s3(bucket, input_key))
    n = len(lines)
    print(f"Retrieved {n} lines.")

    # parallel sentiment analysis
    cores = cpu_count()
    print(f"Running parallel sentiment analysis on {n} lines using {cores} cores...")
    t0 = time.time()
    with Pool(processes=cores) as pool:
        par_sentiments = pool.map(analyze_sentiment, lines)
    t_parallel = time.time() - t0
    par_counts = classify_counts(par_sentiments)
    print(f"Parallel done in {t_parallel:.2f} s:", dict(par_counts))

    # plot parallel bar
    par_chart = f"hybrid_parallel_sentiment_{size}gb.png"
    par_title = f"Parallel Sentiment Analysis\nSize: {size} GB | Time: {t_parallel:.2f} s"
    plot_bar(par_counts, par_title, par_chart)
    boto3.client("s3").upload_file(par_chart, bucket, f"{s3_folder}/{par_chart}")
    print(f"Uploaded parallel chart to s3://{bucket}/{s3_folder}/{par_chart}")

    # sequential sentiment analysis
    print("Running sequential sentiment analysis...")
    t1 = time.time()
    seq_sentiments = [analyze_sentiment(line) for line in lines]
    t_sequential = time.time() - t1
    seq_counts = classify_counts(seq_sentiments)
    print(f"Sequential done in {t_sequential:.2f} s:", dict(seq_counts))

    # plot sequential bar
    seq_chart = f"hybrid_sequential_sentiment_{size}gb.png"
    seq_title = f"Sequential Sentiment Analysis\nSize: {size} GB | Time: {t_sequential:.2f} s"
    plot_bar(seq_counts, seq_title, seq_chart)
    boto3.client("s3").upload_file(seq_chart, bucket, f"{s3_folder}/{seq_chart}")
    print(f"Uploaded sequential chart to s3://{bucket}/{s3_folder}/{seq_chart}")

    # time comparison
    comparison_chart = f"hybrid_sentiment_time_comparison_{size}gb.png"
    speedup = t_sequential / t_parallel if t_parallel > 0 else float('inf')
    comp_title = f"Time Comparison\nSize: {size} GB | Speedup: {speedup:.2f}x"
    times = {'parallel': t_parallel, 'sequential': t_sequential}
    plot_time_comparison(times, comp_title, comparison_chart)
    boto3.client("s3").upload_file(comparison_chart, bucket, f"{s3_folder}/{comparison_chart}")
    print(f"Uploaded comparison chart to s3://{bucket}/{s3_folder}/{comparison_chart}")
