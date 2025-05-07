import boto3
import json
import os
import torch
import io

TOP_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Retrieve AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Create an S3 client using the environment variables
s3_client = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

def fetch_data_from_s3(bucket_name, start: int, end: int):
    """
    Fetch all files from the specified S3 bucket and return their contents in a list.
    """    
    # List all objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    
    if 'Contents' not in response:
        return []

    file_keys = [obj['Key'] for obj in response['Contents']]
    data_list = []

    # Get 10 elements
    file_keys.sort()
    file_keys = file_keys[start:end]

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

def upload_to_s3(pair, file_name):
    # Load into buffer
    buffer = io.BytesIO()
    torch.save(pair, buffer)
    buffer.seek(0)

    try:
        # Upload file name
        s3_client.upload_fileobj(buffer, "sgns-pairs", file_name)
    except Exception as err:
        print("Unable to upload to s3:", str(err))


def get_vocab_size():
    merges_file = TOP_DIRECTORY + "/artifacts/merges.json"
    try:
        with open(merges_file, 'r') as f:
            artifact_map = json.load(f)
        
        # Ordering
        ordering_list = artifact_map["ordering"]
        ordering_pair = ordering_list[len(ordering_list)-1]
        lookup_key = str(ordering_pair[0]) + "," + str(ordering_pair[1])

        # Merges map
        merges_map = artifact_map["merges"]
        vocab_value = merges_map[lookup_key]

        return int(vocab_value) + 1

    except Exception as err:
        print(f"Error reading merges.json: {err}")
        return None

def list_s3_pt_files(bucket_name='sgns-pairs'):
    # List all objects in the bucket
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.pt'):
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size']
                    })
    
    # Sort files by name
    files.sort(key=lambda x: x['key'])
    return files

def fetch_pt_file_from_s3(bucket_name, file_key):
    try:
        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        # Load the object using torch.load
        buffer = io.BytesIO(response['Body'].read())
        return torch.load(buffer)
    except Exception as e:
        print(f"Error fetching or loading {file_key} from S3: {str(e)}")
        return None

def upload_tensor_to_s3(tensor, key):
    # Serialize tensor to in-memory buffer
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)

    # Push to S3
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.upload_fileobj(buffer, 'sgns-artifacts', key)