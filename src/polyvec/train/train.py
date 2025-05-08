import os
import sys
from tqdm import tqdm

# Define the base directory for the project
DATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Set up paths
sys.path.append(DATA_DIRECTORY)
sys.path.append(BASE_DIRECTORY)

from numpy import negative
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from data.sgns import generate_sgns_pairs
import time
import requests
import json
from data.util import get_vocab_size, list_s3_pt_files, fetch_pt_file_from_s3, upload_tensor_to_s3

# Class setup

# Dataset, streamed
class StreamingSGNSDataset(torch.utils.data.IterableDataset):
    def __init__(self, s3_files=None, bucket_name='sgns-pairs-beta'):
        """
        Initialize dataset with files from S3 bucket
        
        Args:
            s3_files (list): List of file metadata from S3, if None will be fetched
            bucket_name (str): Name of the S3 bucket containing the files
        """
        self.bucket_name = bucket_name
        if s3_files is None:
            self.s3_files = list_s3_pt_files(bucket_name)
        else:
            self.s3_files = s3_files
        print(f"Found {len(self.s3_files)} .pt files in S3 bucket {bucket_name}")

    def __iter__(self):
        # Loop through all files in S3
        for file_info in self.s3_files:
            file_key = file_info['key']
            print(f"Loading {file_key} from S3...")
            triplets = fetch_pt_file_from_s3(self.bucket_name, file_key)
            
            if triplets is not None:
                for center, context, negatives in triplets:
                    yield (
                        torch.tensor(center, dtype=torch.long),
                        torch.tensor(context, dtype=torch.long),
                        torch.tensor(negatives, dtype=torch.long),
                    )
            else:
                print(f"Warning: Failed to load {file_key}")


# Embedding Model
class SGNSModel(nn.Module):
    # Initialize embeddings
    def __init__(self, vocab_size, embedding_dimension):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dimension)

    
    # Forward pass, objective function
    def forward(self, center, context, negatives):
        # Get embeddings
        center_embedding = self.input_embedding(center) # (B, D)
        context_embedding = self.output_embedding(context) # (B, D)
        negative_embeddings = self.output_embedding(negatives) # (B, K, D)

        # Hadamard product and sum for (center, context)
        context_affinity = torch.sum(center_embedding * context_embedding, dim=1)

        # Dot product for (center, negatives) --> Unsqueeze then squeeze to make dimensions match
        negative_affinity = torch.bmm(negative_embeddings, torch.unsqueeze(center_embedding, 2)).squeeze(2)

        # Policy
        return -F.logsigmoid(context_affinity).mean() - F.logsigmoid(-negative_affinity).mean()


def train(start_idx, end_idx):    
    # If you need to generate dataset first - if data is present, leave commented out
    # start = time.time()
    # generate_sgns_pairs(start_idx, end_idx)
    # print("Done generating SGNS data", time.time() - start)
    # exit(1)

    # Get vocab size
    vocab_size = get_vocab_size()

    # Set the S3 bucket name containing the .pt files
    s3_bucket_name = 'sgns-pairs'

    # List all .pt files in the S3 bucket
    s3_files = list_s3_pt_files(s3_bucket_name)

    # Set up dataset
    dataset = StreamingSGNSDataset(s3_files, s3_bucket_name)
    cpu_cores = os.cpu_count()
    max_workers = max(1, int(cpu_cores * 0.9))
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=max_workers,
        pin_memory=True,  # Speeds up host to GPU transfers
        prefetch_factor=2,  # Prefetch ahead to keep GPU fed
    )

    # Initialize model and optimizer
    start = time.time()  # Start timer here for epoch tracking
    embedding_dim = 300
    model = SGNSModel(vocab_size, embedding_dim)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Start training loop
    epochs = 5
    
    # Track overall progress
    total_files = len(dataloader.dataset.s3_files) if hasattr(dataloader.dataset, 's3_files') else "unknown"
    processed_files = set()
    
    for i in tqdm(range(epochs), desc="Training epochs"):
        total_loss = 0.0
        count = 0
        
        for batch_idx, (center, context, negatives) in enumerate(tqdm(dataloader, desc=f"Epoch {i+1}/{epochs}", leave=False)):
            # Display batch progress periodically
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx} batches so far in this epoch")
            
            # Convert
            center = center.to(device).long()
            context = context.to(device).long()
            negatives = negatives.to(device).long()

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model(center, context, negatives)
            loss.backward()

            # Set gradients
            optimizer.step()

            # Accrue loss
            total_loss += loss.item()

            # Count batches
            count += 1
        
        # Print statistics for this epoch
        print(f"Epoch: {i+1}/{epochs}")
        print("Average Loss:", total_loss / count)
        print(f"Files processed: {len(processed_files)}/{total_files}")
        print("Elapsed:", time.time() - start)
        print("*" * 100)

        # Save embeddings after each epoch
        upload_tensor_to_s3(
            tensor=model.input_embedding.weight.data,
            key='polyvec_embeddings.pt'
        )


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train(int(sys.argv[1]), int(sys.argv[2]))
