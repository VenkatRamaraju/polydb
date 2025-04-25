import os
import sys

# Define the base directory for the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set up paths
sys.path.append(BASE_DIR)

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

# Class setup

# Dataset, streamed
class StreamingSGNSDataset(torch.utils.data.IterableDataset):
    def __init__(self, shard_paths):
        self.shard_paths = shard_paths

    def __iter__(self):
        # Loop through all shards
        for shard_path in self.shard_paths:
            triplets = torch.load(shard_path)
            for center, context, negatives in triplets:
                yield (
                    torch.tensor(center, dtype=torch.long),
                    torch.tensor(context, dtype=torch.long),
                    torch.tensor(negatives, dtype=torch.long),
                )


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

        # Dot product for (center, negatives) --> Squeeze/Unsqueeze to make dimensions match
        negative_affinity = torch.bmm(negative_embeddings, torch.unsqueeze(center_embedding, 2)).squeeze(2)

        # Calculate and return loss
        return -F.logsigmoid(context_affinity).mean() - F.logsigmoid(-negative_affinity).mean()


# Training loop

# If you need to generate dataset first - if data is present, leave commented out
start = time.time()
generate_sgns_pairs()
print("Done generating SGNS data", time.time() - start)


# Get vocab size
response = requests.get("http://localhost:8080/vocabulary-size")
response_content = response.content.decode('utf-8')
vocab_data = json.loads(response_content)
vocab_size = vocab_data["vocabulary_size"]


# List all files in the artifacts/pairs directory
pairs_directory = os.path.join(BASE_DIR, 'artifacts', 'pairs')
pair_files = sorted([
    os.path.join(pairs_directory, f)
    for f in os.listdir(pairs_directory)
    if f.endswith('.pt')
])

# Set up dataset
dataset = StreamingSGNSDataset(pair_files)
dataloader = DataLoader(
    dataset,
    batch_size=512,
    num_workers=4,
    pin_memory=True,
    # no shuffle!
)

# Initialize model and optimizer
embedding_dim = 300
model = SGNSModel(vocab_size, embedding_dim)
optimizer = Adam(model.parameters(), lr=1e-3)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Start training loop
epochs = 5
for i in range(epochs):
    total_loss = 0.0
    for center, context, negatives in dataloader:
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
    
    # Print statistics for this epoch
    print("Epoch:", i)
    print("Average Loss:", total_loss / len(dataloader))
    print("Elapsed:", time.time() - start)
    print("*" * 100)

    # Save embeddings after each epoch
    torch.save(model.input_embedding.weight.data, os.path.join(BASE_DIR, 'artifacts', 'polyvec_embeddings_' + str(i) + '.pt'))
