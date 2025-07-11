from multiprocessing import Pool, set_start_method, cpu_count
from collections import Counter
import matplotlib.pyplot as plt
import tempfile
import boto3
import argparse
import time
import os

# Use fork start method on Linux
try:
    set_start_method("fork")
except RuntimeError:
    pass

def count_words(lines):
    """Count words in a list of lines."""
    counter = Counter()
    for line in lines:
        counter.update(line.strip().lower().split())
    return counter

def split_into_chunks(data, num_chunks):
    """Split a list into num_chunks roughly equal parts."""
    chunk_size = len(data) // num_chunks
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Word Count from S3")
    parser.add_argument(
        "--filesize",
        type=float,
        required=True,
        help="File size identifier: 0.5, 1.0, 1.5, etc."
    )
    args = parser.parse_args()

    bucket_name = "scalable-project-group2"
    s3 = boto3.client("s3")

    if args.filesize == 0.5:
        s3_input_key = "text_files/test_file_05.txt"
    elif args.filesize == 1.0:
        s3_input_key = "text_files/test_file_10.txt"
    elif args.filesize == 1.5:
        s3_input_key = "text_files/test_file_15.txt"
    else:
        print("Unsupported file size.")
        exit(1)

    # Download input file from S3
    print("Downloading file from S3...")
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmp_file:
        s3.download_fileobj(bucket_name, s3_input_key, tmp_file)
        tmp_file_path = tmp_file.name

    # Read file into memory
    print("Reading file into memory...")
    with open(tmp_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    num_cores = cpu_count()
    print(f"Using {num_cores} cores for processing...")
    chunks = split_into_chunks(lines, num_cores)

    # Parallel word count
    start = time.time()
    with Pool(processes=num_cores) as pool:
        results = pool.map(count_words, chunks)
    total_counter = sum(results, Counter())
    end = time.time()

    # Output statistics
    top_words = total_counter.most_common(10)
    print("Parallel word count completed")
    print("Top 10 words:", top_words)
    print("Total unique words:", len(total_counter))
    print("Total word count:", sum(total_counter.values()))
    print("Time taken (s):", round(end - start, 2))

    # Plot results
    print("Creating chart...")
    words, counts = zip(*top_words)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(words, counts, edgecolor="black")

    max_val = max(counts)
    for bar in bars:
        height = bar.get_height()
        padding = max_val * 0.01
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + padding,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.title(f"Top 10 Parallel Word Frequencies {args.filesize} gb  (Time: {round(end - start, 2)} sec)")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    chart_filename = f"parallel_word_count_{args.filesize}.png"
    s3_output_key = f"word_count_graphs/parallel/word_count_top10_{args.filesize}gb.png"
    plt.savefig(chart_filename)

    # Upload chart to S3
    print("Uploading chart to S3...")
    try:
        s3.upload_file(chart_filename, bucket_name, s3_output_key)
        print(f"Uploaded to s3://{bucket_name}/{s3_output_key}")
    except Exception as e:
        print("S3 upload failed:", e)
