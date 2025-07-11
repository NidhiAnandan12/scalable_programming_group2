from multiprocessing import Pool, cpu_count, set_start_method
from textblob import TextBlob
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import boto3
import tempfile
import argparse
import time

try:
    set_start_method("fork")
except RuntimeError:
    pass  # already set

def download_from_s3(bucket: str, key: str) -> str:
    """Download an S3 object into a temp file and return its path."""
    s3 = boto3.client("s3")
    tmp = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
    s3.download_fileobj(bucket, key, tmp)
    tmp.close()
    return tmp.name

def analyze_sentiment(line: str) -> float:
    """Return the polarity of a single line of text."""
    return TextBlob(line).sentiment.polarity

def main():
    parser = argparse.ArgumentParser(description="Parallel Sentiment Analysis from S3")
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
        return

    input_key = key_map[size]
    output_chart_key = f"sentiment_analysis/parallel/parallel_sentiment_analysis_{size}gb.png"

    print("Downloading input file from S3...")
    local_path = download_from_s3(bucket, input_key)

    with open(local_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    num_cores = cpu_count()
    print(f"Using {num_cores} cores for sentiment analysis...")

    start = time.time()
    sentiments = []
    with Pool(processes=num_cores) as pool:
        for polarity in tqdm(
            pool.imap_unordered(analyze_sentiment, lines),
            total=len(lines),
            desc="Analyzing"
        ):
            sentiments.append(polarity)
    elapsed = time.time() - start

    positive = sum(1 for s in sentiments if s > 0)
    negative = sum(1 for s in sentiments if s < 0)
    neutral  = sum(1 for s in sentiments if s == 0)
    total_reviews  = len(sentiments)
    average_score  = sum(sentiments) / total_reviews

    print("Sentiment analysis complete")
    print(f"Average sentiment score: {average_score:.4f}")
    print(f"Positive reviews: {positive}")
    print(f"Negative reviews: {negative}")
    print(f"Neutral reviews: {neutral}")
    print(f"Total reviews: {total_reviews}")
    print(f"Time taken: {elapsed:.2f} seconds")

    # save summary to a text file
    with open("sentiment_output_parallel.txt", "w", encoding="utf-8") as out:
        out.write(f"Average sentiment score: {average_score:.4f}\n")
        out.write(f"Positive reviews: {positive}\n")
        out.write(f"Negative reviews: {negative}\n")
        out.write(f"Neutral reviews: {neutral}\n")
        out.write(f"Total reviews: {total_reviews}\n")
        out.write(f"Time taken: {elapsed:.2f} seconds\n")

    # prepare data for bar chart
    counts = {"positive": positive, "negative": negative, "neutral": neutral}

    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.keys(), counts.values(), edgecolor="black")
    max_val = max(counts.values())
    for bar in bars:
        h = bar.get_height()
        padding = max_val * 0.01
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + padding,
            str(int(h)),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    # include process name, size, and time in the title
    plt.title(f"Parallel Sentiment Analysis (Size: {size} GB, Time: {elapsed:.2f} s)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()

    chart_filename = f"parallel_sentiment_analysis_{size}GB.png"
    plt.savefig(chart_filename)
    plt.close()

    # upload the chart back to S3
    boto3.client("s3").upload_file(chart_filename, bucket, output_chart_key)
    print(f"Uploaded chart to s3://{bucket}/{output_chart_key}")

if __name__ == "__main__":
    main()
