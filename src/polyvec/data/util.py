import boto3
import json
import os


def fetch_data_from_s3(bucket_name='tknzr', region_name='us-east-1'):
    """
    Fetch all files from the specified S3 bucket and return their contents in a list.
    """
    # Retrieve AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # Create an S3 client using the environment variables
    s3_client = boto3.client(
        's3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    # List all objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    
    if 'Contents' not in response:
        return []

    file_keys = [obj['Key'] for obj in response['Contents']]
    data_list = []

    # Get 10 elements
    file_keys = file_keys[:1]

    # Fetch each file
    for file_key in file_keys:
        print(file_key)
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = obj['Body'].read().decode('utf-8')
        try:
            # Load file
            data = json.loads(file_content)
            
            # Add just the sentences
            for language in data.keys():
                data_list.extend(data[language])
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_key}")

    return data_list
    