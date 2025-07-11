from multiprocessing import Pool, set_start_method
from collections import Counter, deque
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import boto3
import time
import os

# Set multiprocessing start method (Linux/Unix)
try:
    set_start_method("fork")
except RuntimeError:
    pass

# ---------- CONFIG ----------
REGION = 'us-east-1'
STREAM_NAME = '23286873_kinesis'
BUCKET_NAME = 'x23286873scalables3bucket'
BATCH_SIZE = 100
NUM_WORKERS = 2
CHUNK_SIZE = 25000
POLL_INTERVAL = 10  # seconds
WINDOW_DURATION = timedelta(minutes=5)
# ----------------------------

# Store (timestamp, Counter) tuples for sliding window
sliding_window = deque()

# ---------- Helper Functions ----------
def count_words_batch(lines):
    counter = Counter()
    for line in lines:
        counter.update(line.strip().lower().split())
    return counter

def chunkify_memory(lines, chunk_size=CHUNK_SIZE):
    for i in range(0, len(lines), chunk_size):
        yield lines[i:i + chunk_size]

def process_lines(lines):
    chunks = list(chunkify_memory(lines))
    with Pool(NUM_WORKERS) as pool:
        counters = pool.map(count_words_batch, chunks)

    combined = Counter()
    for c in counters:
        combined.update(c)
    return combined

def update_sliding_window(word_counts):
    now = datetime.now()
    sliding_window.append((now, word_counts))

    # Remove old records beyond window
    cutoff = now - WINDOW_DURATION
    while sliding_window and sliding_window[0][0] < cutoff:
        sliding_window.popleft()

def get_trending_words():
    total = Counter()
    for _, wc in sliding_window:
        total.update(wc)
    return total.most_common(5)

def plot_and_upload(top_words, s3, timestamp):
    words, counts = zip(*top_words)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(words, counts, edgecolor='black')

    max_val = max(counts)
    for bar in bars:
        height = bar.get_height()
        padding = max_val * 0.01
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + padding,
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black',
            fontweight='bold'
        )

    plt.title(f"Top 5 Trending Words 5minute")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    chart_filename = f"top5_words.png"

    # Optional: remove old file if exists
    if os.path.exists(chart_filename):
        os.remove(chart_filename)

    # Save chart image
    plt.savefig(chart_filename)
    plt.close()

    # Upload to S3
    s3_output_key = f"word_count_graphs/streaming/{chart_filename}"
    try:
        s3.upload_file(chart_filename, BUCKET_NAME, s3_output_key)
        print(f"✅ Uploaded chart to s3://{BUCKET_NAME}/{s3_output_key}")
    except Exception as e:
        print("❌ S3 upload failed:", e)

    # Clean up local file
    os.remove(chart_filename)
# ---------------------------------------

# ---------- Main Kinesis Logic ----------
def main():
    print("Connecting to Kinesis...")
    kinesis = boto3.client('kinesis', region_name=REGION)
    s3 = boto3.client('s3', region_name=REGION)

    # Get Shard ID
    stream_desc = kinesis.describe_stream(StreamName=STREAM_NAME)
    shard_id = stream_desc['StreamDescription']['Shards'][0]['ShardId']

    # Get Shard Iterator
    shard_iter = kinesis.get_shard_iterator(
        StreamName=STREAM_NAME,
        ShardId=shard_id,
        ShardIteratorType='LATEST'
    )['ShardIterator']

    print("Waiting for new records...")

    while True:
        response = kinesis.get_records(ShardIterator=shard_iter, Limit=BATCH_SIZE)
        shard_iter = response['NextShardIterator']

        if response['Records']:
            lines = [record['Data'].decode('utf-8') for record in response['Records']]
            print(f"\nReceived {len(lines)} new records")

            word_counts = process_lines(lines)
            update_sliding_window(word_counts)

            top_words = get_trending_words()
            print("Top 5 Trending Words (Last 5 Minute):")
            for word, count in top_words:
                print(f"{word}: {count}")

            if top_words:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_and_upload(top_words, s3, timestamp)
        else:
            print("No new records")

        time.sleep(POLL_INTERVAL)

# ---------------------------------------

if __name__ == "__main__":
    main()