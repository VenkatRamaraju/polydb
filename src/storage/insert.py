import faiss
import pickle
import os
import numpy as np
import hashlib
import sys
import torch

# Define base path
BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIRECTORY)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'polyvec')))

from polyvec.train.embeddings import generate_embeddings

# Paths
INDEX_PATH = BASE_DIRECTORY + "/storage/faiss.index"
METADATA_PATH = BASE_DIRECTORY + "/storage/metadata.pkl"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize index and metadata
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    # IndexIDMap so that we can store UUID mappings
    dimension = 300 
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

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
    print(f"Saved index to {INDEX_PATH}")

# Hash
def hash(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')

# Insert new embeddings
def insert_embedding(text, embeddings, uuid):
    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Reshape for storage
    embeddings = embeddings.astype(np.float32).reshape(1, -1)

    # Store embeddings in a persistent index using FAISS
    uuid_int = hash(uuid)
    
    # Add to index
    index.add_with_ids(embeddings, np.array([uuid_int], dtype=np.int64))
    
    # Store mapping
    metadata[uuid_int] = text
    
    # Persist on disk
    persist()


def find_similar_vectors(query_embedding, top_k=5):
    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.detach().cpu().numpy()

    # Reshape for search
    query_vector = query_embedding.astype(np.float32).reshape(1, -1)
    
    # Search the index
    _, ids = index.search(query_vector, top_k)

    # Return texts from mapping
    return [metadata[id] for id in ids[0] if id != -1]