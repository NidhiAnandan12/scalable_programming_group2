import argparse
import tempfile
import time
from collections import Counter

import boto3
import matplotlib.pyplot as plt


def download_from_s3(bucket: str, key: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
    boto3.client("s3").download_fileobj(bucket, key, tmp)
    tmp.close()
    return tmp.name


def sequential_word_count(file_path: str) -> Counter:
    counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            counter.update(line.strip().lower().split())
    return counter


def save_full_counts(counter: Counter, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("word,count\n")
        for w, c in counter.items():
            f.write(f"{w},{c}\n")


# 1) Make sure elapsed is in the signature here
def plot_top10_and_upload(counter: Counter,
                          bucket: str,
                          key: str,
                          size_tag: float,
                          elapsed: float):
    top10 = counter.most_common(10)
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

    # 2) Include elapsed in the title f-string
    plt.title(f"Top 10 Word Frequencies (Size: {size_tag} GB, Time: {elapsed:.2f} s)")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    local_chart = f"sequential_word_count_top10_{size_tag}GB.png"
    plt.savefig(local_chart)
    plt.close()

    boto3.client("s3").upload_file(local_chart, bucket, key)
    print(f"Chart uploaded to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser("Sequential Word Count from S3")
    parser.add_argument("--filesize", type=float, required=True,
                        help="0.5, 1.0, 1.5, etc.")
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
        return

    in_key = key_map[size]
    out_key = f"word_count_graphs/sequential/word_count_top10_{size}gb.png"

    print("Downloading...")
    path = download_from_s3(bucket, in_key)

    print("Counting words...")
    t0 = time.time()
    cnt = sequential_word_count(path)
    elapsed = time.time() - t0

    print("Top 10:", cnt.most_common(10))
    print("Unique words:", len(cnt))
    print(f"Time taken: {elapsed:.2f} s")

    print("Saving full counts...")
    save_full_counts(cnt, "wordcount_sequential_output.txt")

    # 3) Pass elapsed into the call here
    print("Plotting & uploading chart...")
    plot_top10_and_upload(cnt, bucket, out_key, size, elapsed)


if __name__ == "__main__":
    main()
