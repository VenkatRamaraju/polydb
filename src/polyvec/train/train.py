import os
import sys

# Define the base directory for the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Set up paths
sys.path.append(os.path.join(BASE_DIR, 'src'))

from numpy import negative
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from data.sgns import generate_sgns_pairs
import time


# Class setup

# Dataset class
class SGNSDataset(Dataset):
    def __init__(self, triplets):
        super().__init__()
        self.pairs = triplets

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        center, context, negatives = self.pairs[index]
        return (
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

# Initialize dataset and dataloader
start = time.time()
training_triplets, vocab_size = generate_sgns_pairs()
print("Done with data", time.time() - start)
dataset = SGNSDataset(training_triplets)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

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
        center = center.long()
        context = context.long()
        negatives = negatives.long()

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        loss = model(center, context, negatives)
        loss.backward()

        # Set gradients
        optimizer.step()

        # Accrue loss
        total_loss += loss.item()
    
    print("Epoch", i, "Loss", total_loss, "Elapsed", time.time() - start)


# Save input embeddings
torch.save(model.input_embedding.weight.data, os.path.join(BASE_DIR, 'artifacts', 'polyvec_embeddings.pt'))
