import faiss
import pickle
import os
import numpy as np
import sys
import torch
import uuid

# Define base path
BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'polyvec')))
sys.path.append(BASE_DIRECTORY)

from polyvec.train.embeddings import generate_embeddings

# Paths
INDEX_PATH = BASE_DIRECTORY + "/artifacts/faiss.index"
METADATA_PATH = BASE_DIRECTORY + "/artifacts/metadata.pkl"

# Initialize index and metadata
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    # IndexIDMap so that we can store UUID mappings
    DIMENSION = 300
    index = faiss.IndexIDMap(faiss.IndexFlatL2(DIMENSION))

# Initialize metadata
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
else:
    metadata = {}

def persist():
    # Save metadata
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save index
    faiss.write_index(index, INDEX_PATH)

# Convert UUID string to a positive integer (compatible with FAISS)
def uuid_to_int(uuid_str):
    # Parse UUID string to UUID object if it's not already one
    if isinstance(uuid_str, str):
        uuid_obj = uuid.UUID(uuid_str)
    else:
        uuid_obj = uuid_str
        
    # Take the first 31 bits (to ensure it's a positive int within C long range)
    return uuid_obj.int & 0x7FFFFFFF

# Insert new embeddings
def insert_embedding(text, embeddings, uuid_str):
    # Convert to correct dimension with mean pooling
    embeddings = embeddings.mean(axis=0)

    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Reshape for storage
    embeddings = embeddings.astype(np.float32).reshape(1, -1)

    # Store embeddings in a persistent index using FAISS
    uuid_int = uuid_to_int(uuid_str)
    
    # Add to index
    index.add_with_ids(embeddings, np.array([uuid_int], dtype=np.int64))
    
    # Store mapping
    metadata[uuid_int] = text
    
    # Persist on disk
    persist()


def find_similar_embeddings(query_embedding, top_k=5):
    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.detach().cpu().numpy()

    # Convert to correct dimension with mean pooling
    query_embedding = query_embedding.mean(axis=0)

    # Reshape for search
    query_vector = query_embedding.astype(np.float32).reshape(1, -1)
    
    # Search the index
    _, ids = index.search(query_vector, top_k)

    # Return texts from mapping
    return [metadata[id] for id in ids[0] if id != -1]
