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

# If unable to load from S3, create dummy embedding matrix for testing
if embedding_matrix is None:
    print("Creating a dummy embedding matrix for testing (10000 x 300)")
    embedding_matrix = torch.rand((10000, 300))  # Dummy matrix with random values

def generate_embeddings(token_ids):
    # Convert token_ids to a tensor and limit to valid indices
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    # Clamp indices to be within the valid range
    token_tensor = torch.clamp(token_tensor, 0, embedding_matrix.size(0) - 1)
    return embedding_matrix[token_tensor]
