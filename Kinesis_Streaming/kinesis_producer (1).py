import boto3
import time

# AWS configuration
region = 'us-east-1'
bucket_name = 'x23286873scalables3bucket'
file_key = 'test_text_files/test_file_15.txt'
stream_name = '23286873_kinesis'

# Create clients
s3 = boto3.client('s3', region_name=region)
kinesis = boto3.client('kinesis', region_name=region)

# Read file content from S3
print("Fetching file from S3...")
obj = s3.get_object(Bucket=bucket_name, Key=file_key)
lines = obj['Body'].read().decode('utf-8').splitlines()

# Send each line to Kinesis
print("Streaming to Kinesis...")
for line in lines:
    kinesis.put_record(
        StreamName=stream_name,
        Data=line,
        PartitionKey='partitionkey'
    )
    print(f"Sent: {line}")
    time.sleep(1)
print("done")