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
    body = resp["Body"]
    for raw in body.iter_lines(chunk_size=1024):
        yield raw.decode("utf-8", errors="ignore")


def split_into_chunks(data, num_chunks):
    """
    Split a list into num_chunks roughly equal parts.
    """
    chunk_size = len(data) // num_chunks
    return [data[i * chunk_size:(i + 1) * chunk_size]
            for i in range(num_chunks)]


def count_words(lines):
    """
    Count words in a list of lines (for parallel processing).
    """
    counter = Counter()
    for line in lines:
        counter.update(line.strip().lower().split())
    return counter


def sequential_word_count(lines):
    """
    Count words by iterating through each line (sequential).
    """
    counter = Counter()
    for line in lines:
        counter.update(line.strip().lower().split())
    return counter


def plot_top10(counter: Counter, title: str, out_path: str):
    """
    Plot the top 10 most common words from `counter`, save to out_path.
    """
    top10 = counter.most_common(10)
    if not top10:
        return
    words, counts = zip(*top10)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(words, counts, edgecolor="black")
    max_count = max(counts)
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_count * 0.01,
            str(int(h)),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_time_comparison(times: dict, title: str, out_path: str):
    """
    Plot a line chart comparing times (keys: labels, values: seconds).
    """
    labels = list(times.keys())
    secs = [times[label] for label in labels]

    plt.figure(figsize=(6, 4))
    plt.plot(labels, secs, marker="o")
    for x, y in zip(labels, secs):
        plt.text(x, y, f"{y:.2f}s", ha="center", va="bottom", fontweight="bold")

    plt.title(title)
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    # use fork on Linux
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Hybrid Word Count Directly from S3")
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
    s3_folder = "word_count_graphs/hybrid"

    print("Streaming lines directly from S3...")
    lines = list(stream_lines_from_s3(bucket, input_key))
    total_lines = len(lines)
    print(f"Retrieved {total_lines} lines.")

    # Parallel word count
    cores = cpu_count()
    print(f"Running parallel word count using {cores} cores...")
    chunks = split_into_chunks(lines, cores)
    t0 = time.time()
    with Pool(processes=cores) as pool:
        partials = pool.map(count_words, chunks)
    parallel_counter = sum(partials, Counter())
    t_parallel = time.time() - t0
    print(f"Parallel done in {t_parallel:.2f}s.")

    par_chart = f"hybrid_parallel_top10_{size}GB.png"
    par_title = f"Parallel Word Count\nSize: {size} GB | Time: {t_parallel:.2f} s"
    plot_top10(parallel_counter, par_title, par_chart)
    boto3.client("s3").upload_file(par_chart, bucket, f"{s3_folder}/{par_chart}")
    print("Uploaded parallel chart.")

    # Sequential word count
    print("Running sequential word count...")
    t1 = time.time()
    sequential_counter = sequential_word_count(lines)
    t_seq = time.time() - t1
    print(f"Sequential done in {t_seq:.2f}s.")

    seq_chart = f"hybrid_sequential_top10_{size}GB.png"
    seq_title = f"Sequential Word Count\nSize: {size} GB | Time: {t_seq:.2f} s"
    plot_top10(sequential_counter, seq_title, seq_chart)
    boto3.client("s3").upload_file(seq_chart, bucket, f"{s3_folder}/{seq_chart}")
    print("Uploaded sequential chart.")

    # Time-comparison
    comparison_chart = f"hybrid_time_comparison_{size}GB.png"
    speedup = t_seq / t_parallel if t_parallel else float("inf")
    comp_title = f"Time Comparison\nSize: {size} GB | Speedup: {speedup:.2f}x"
    plot_time_comparison(
        { "Parallel": t_parallel,"Sequential": t_seq},
        comp_title,
        comparison_chart
    )
    boto3.client("s3").upload_file(comparison_chart, bucket, f"{s3_folder}/{comparison_chart}")
    print("Uploaded comparison chart.")
