import torch
import os
import sys
from data.util import fetch_pt_file_from_s3

# Define base path
BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Set up paths
sys.path.append(DATA_DIRECTORY)
sys.path.append(BASE_DIRECTORY)

# More imports
from sgns import process_sentence

# Load embeddings globally at startup
embedding_matrix = fetch_pt_file_from_s3("sgns-artifacts", "polyvec_embeddings.pt")

def generate_embeddings(token_ids):
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)

    return embedding_matrix[token_ids]
