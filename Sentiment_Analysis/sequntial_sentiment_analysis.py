import argparse
import tempfile
import time
from collections import Counter

from textblob import TextBlob
from tqdm import tqdm
import matplotlib.pyplot as plt
import boto3

def download_from_s3(bucket: str, key: str) -> str:
    """Download an S3 object into a temp file and return its path."""
    s3 = boto3.client("s3")
    tmp = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
    s3.download_fileobj(bucket, key, tmp)
    tmp.close()
    return tmp.name

def analyze_sentiment(line: str) -> float:
    """Return the polarity score of a single line using TextBlob."""
    return TextBlob(line).sentiment.polarity

def main():
    parser = argparse.ArgumentParser(description="Sequential Sentiment Analysis from S3")
    parser.add_argument(
        "--filesize",
        type=float,
        required=True,
        help="File size identifier: 0.5, 1.0, 1.5, etc."
    )
    args = parser.parse_args()

    bucket = "scalable-project-group2"
    size = args.filesize

    # map the filesize to its S3 key
    key_map = {
        0.5: "text_files/test_file_05.txt",
        1.0: "text_files/test_file_10.txt",
        1.5: "text_files/test_file_15.txt",
    }
    if size not in key_map:
        print("Unsupported file size.")
        return

    input_key = key_map[size]
    output_key = f"sentiment_analysis/sequential/sequential_sentiment_{size}gb.png"

    print("Downloading input file from S3...")
    local_path = download_from_s3(bucket, input_key)

    # Read all lines
    with open(local_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Analyzing {len(lines)} lines sequentially with TextBlob...")
    start = time.time()
    sentiments = [
        analyze_sentiment(line)
        for line in tqdm(lines, desc="Sentiment")
    ]
    elapsed = time.time() - start

    # Aggregate counts
    positive = sum(1 for s in sentiments if s > 0)
    negative = sum(1 for s in sentiments if s < 0)
    neutral  = sum(1 for s in sentiments if s == 0)
    total    = len(sentiments)
    avg_score = sum(sentiments) / total if total else 0.0

    print("Sequential Sentiment Analysis Complete")
    print(f"Average Sentiment Score: {avg_score:.4f}")
    print(f"Positive: {positive}, Negative: {negative}, Neutral: {neutral}")
    print(f"Total Lines: {total}")
    print(f"Time Taken: {elapsed:.2f} seconds")

    # Plot results
    counts = {"positive": positive, "negative": negative, "neutral": neutral}
    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.keys(), counts.values(), edgecolor="black")
    max_val = max(counts.values())
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + max_val*0.01,
            str(int(h)),
            ha="center", va="bottom", fontweight="bold"
        )

    title = (
        f"Sequential Sentiment Analysis\n"
        f"Size: {size} GB | Time: {elapsed:.2f} s"
    )
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()

    chart_file = f"sequential_sentiment_{size}gb.png"
    plt.savefig(chart_file)
    plt.close()

    # Upload chart to S3
    print("Uploading chart to S3...")
    boto3.client("s3").upload_file(chart_file, bucket, output_key)
    print(f"Chart available at s3://{bucket}/{output_key}")

if __name__ == "__main__":
    main()
