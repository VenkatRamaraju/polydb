from .util import fetch_data_from_s3, upload_to_s3
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

# Define the base directory for the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

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


def process_chunk(chunk, chunk_index, vocab_size, neg_sampling_probs, window_size, negative_sample_size):
    token_pairs = []
    for tokens in chunk:
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
    file_name = "chunk_" + str(chunk_index) + ".pt"
    upload_to_s3(token_pairs, file_name)


def process_sentence(sentence):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with requests.Session() as session:
                response = session.post("http://localhost:8080/encode", json=sentence)
                
                if response.status_code == 200:
                    json_response = response.json()
                    tokens = json_response.get("tokens", [])
                    if not tokens:
                        print("Unprocessable request:", sentence, json_response)
                    return tokens
                else:
                    print(f"Attempt {attempt + 1}: Failed to encode sentence, Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Request failed with exception: {e}")
        
        # Optional: Add a delay between retries
        time.sleep(1)
    
    print("All retry attempts failed.")
    return None


def generate_sgns_pairs(start_idx, end_idx):
    # Grab data
    start = time.time()
    sentences = fetch_data_from_s3(start_idx, end_idx)
    print("Done grabbing data from S3", time.time() - start)

    # Process sentences in parallel
    token_freqs = Counter()
    token_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_sentence, sentence): sentence for sentence in sentences}
        for future in concurrent.futures.as_completed(futures):
            tokens = future.result()
            if tokens:
                token_list.append(tokens)

    print("Done getting tokens", time.time() - start)

    # Get frequencies
    token_freqs = Counter()
    for sentence in token_list:
        token_freqs.update(sentence)

    # Get vocab size
    response = requests.get("http://localhost:8080/vocabulary-size")
    response_content = response.content.decode('utf-8')
    vocab_data = json.loads(response_content)
    vocab_size = vocab_data["vocabulary_size"]

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
    chunk_size = 100000

    print("Ready to start processing chunks", time.time() - start)

    # Use ThreadPoolExecutor for parallel processing
    print("Total token list size", len(token_list))
    print("Batch size", len(token_list) // chunk_size)

    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for chunk_index in range(0, len(token_list), chunk_size):
            chunk = token_list[chunk_index:chunk_index + chunk_size]
            print("Chunk index", chunk_index // chunk_size)
            futures.append(executor.submit(process_chunk, chunk, start + (chunk_index // chunk_size), vocab_size, neg_sampling_probs, window_size, negative_sample_size))

        # Wait for all futures to complete
        concurrent.futures.wait(futures)
    
    print("Done processing chunks", time.time() - start)
