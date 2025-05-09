import requests
from collections import Counter
import numpy as np
import json
import random
import torch
import concurrent.futures
import time
import os
import requests
import time
import grpc
import sys
from tqdm import tqdm
from datetime import datetime

# Sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'proto')))

from util import fetch_data_from_s3, upload_to_s3, get_vocab_size
import tokenizerpb.tokenizer_pb2 as tokenizer_pb2
import tokenizerpb.tokenizer_pb2_grpc as tokenizer_pb2_grpc

# Create a GRPC client
class TokenizerClient:
    def __init__(self, socket_path="/tmp/tokenizer.sock"):
        self.channel = grpc.insecure_channel(f'unix://{socket_path}')
        self.stub = tokenizer_pb2_grpc.TokenizerStub(self.channel)

    def encode(self, sentence):
        request = tokenizer_pb2.EncodeRequest(text=sentence)
        return self.stub.Encode(request)

# Instantiate the GRPC client
client = TokenizerClient()

def sample_negatives(k, vocab_size, sampling_probs, forbidden):
    negatives = []
    while len(negatives) < k:
        candidates = np.random.choice(vocab_size, size=k, p=sampling_probs)
        for c in candidates:
            if c not in forbidden:
                negatives.append(c)
                if len(negatives) == k:
                    break
    return negatives


def process_chunk(chunk, file_name, vocab_size, neg_sampling_probs, window_size, negative_sample_size):
    token_pairs = []
    for tokens in tqdm(chunk):
        # Build window
        each_side = window_size // 2

        # Iterate over every sentence
        for i, token in enumerate(tokens):
            # Seen
            seen = set()

            # Left window
            left = i-1
            while left >= 0 and i - left <= each_side:
                seen.add(tokens[left])
                left -= 1

            # Right window
            right = i+1
            while right < len(tokens) and right - i <= each_side:
                seen.add(tokens[right])
                right += 1

            # Negative sampling
            context = list(seen)

            # Add current token to seen list, not context list
            seen.add(token)

            # Create SGNS pairs
            for context_token in context:
                negative_samples = [int(num) for num in sample_negatives(negative_sample_size, vocab_size, neg_sampling_probs, seen)]
                token_pairs.append((token, context_token, negative_samples))

    # Upload to s3
    upload_to_s3(token_pairs, file_name)


def process_sentence(sentence):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.encode(sentence)
            return response.tokens
        except grpc.RpcError as e:
            print(f"Attempt {attempt + 1}: gRPC call failed with exception: {e}")
        
        time.sleep(1)  # Optional delay between retries

    print("All retry attempts failed.")
    return None


def generate_sgns_pairs(start_idx, end_idx):
    # Grab data
    start_time = time.time()
    sentences = fetch_data_from_s3("tknzr", start_idx, end_idx)
    print("Done grabbing data from S3", time.time() - start_time)

    # Process sentences in parallel
    cpu_cores = os.cpu_count()
    max_workers = max(1, int(cpu_cores * 0.75))
    token_freqs = Counter()
    token_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_sentence, sentence): sentence for sentence in sentences}
        with tqdm(total=len(futures), desc="Processing sentences") as pbar:
            for future in concurrent.futures.as_completed(futures):
                tokens = future.result()
                if tokens:
                    token_list.append(tokens)
                pbar.update(1)

    print("Done getting tokens", time.time() - start_time)

    # Get frequencies
    token_freqs = Counter()
    for sentence in token_list:
        token_freqs.update(sentence)

    # Get vocab size
    vocab_size = get_vocab_size()

    # Initialize an array for probabilities
    freq_array = np.zeros(vocab_size, dtype=np.float64)

    # Fill in frequencies
    for token_id, freq in token_freqs.items():
        freq_array[token_id] = freq

    # Soft correction
    neg_sampling_probs = freq_array ** 0.75
    neg_sampling_probs /= neg_sampling_probs.sum()

    # Iterate through sentences in chunks
    window_size = 5
    negative_sample_size = 15
    chunk_size = 10000

    # Use ThreadPoolExecutor for parallel processing
    print("Ready to start processing chunks", time.time() - start_time)
    print("Total token list size", len(token_list))
    print("Batch size", len(token_list) // chunk_size)

    # Process chunks in parallel
    cpu_cores = os.cpu_count()
    max_workers = max(1, int(cpu_cores * 0.75))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunk_index in range(0, len(token_list), chunk_size):
            # Get a single chunk
            chunk = token_list[chunk_index:chunk_index + chunk_size]

            # Inside your function where you define the file_name
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            # print("Chunk number", chunk_index // chunk_size)
            file_name = f"{start_idx}_{chunk_index // chunk_size}_{current_time}.pt"
            futures.append(executor.submit(process_chunk, chunk, file_name, vocab_size, neg_sampling_probs, window_size, negative_sample_size))

        # Wait for all futures to complete with progress tracking
        with tqdm(total=len(futures), desc="Processing chunks") as pbar:
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Get the result to catch any exceptions
                pbar.update(1)
    
    print("Done processing chunks", time.time() - start_time)
