from .util import fetch_data_from_s3
import requests
from collections import Counter
import numpy as np
import json
import random


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


def generate_sgns_pairs():
    # Grab data
    sentences = fetch_data_from_s3()
    print("Done grabbing data...")

    # Grab and count frequencies of tokens
    token_freqs = Counter()
    token_list = []
    for sentence in sentences:
        response = requests.post("http://localhost:8080/encode", json=sentence)
        if response.status_code == 200:
            tokens = response.json().get("tokens", [])
            token_list.append(tokens)
            token_freqs.update(tokens)
        else:
            print(f"Failed to encode sentence: {sentence}, Status Code: {response.status_code}")


    # Get vocab size
    response = requests.get("http://localhost:8080/vocabulary-size")
    response_content = response.content.decode('utf-8')
    vocab_data = json.loads(response_content)
    vocab_size = vocab_data["vocabulary_size"]

    # Get frequencies
    token_freqs = Counter()
    for sentence in token_list:
        token_freqs.update(sentence)

    # Initialize an array for probabilities
    freq_array = np.zeros(vocab_size, dtype=np.float64)

    # Fill in frequencies
    for token_id, freq in token_freqs.items():
        freq_array[token_id] = freq

    # Soft correction
    neg_sampling_probs = freq_array ** 0.75
    neg_sampling_probs /= neg_sampling_probs.sum()

    # Iterate through sentences
    window_size = 5
    negative_sample_size = 15
    token_pairs = []
    for tokens in token_list:
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

    # Random shuffle
    random.shuffle(token_pairs)

    return token_pairs, vocab_size