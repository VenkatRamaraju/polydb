import faiss
import pickle
import os
import numpy as np
import hashlib
import sys
import torch

# Define base path
BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths
INDEX_PATH = BASE_DIRECTORY + "/storage/faiss.index"
METADATA_PATH = BASE_DIRECTORY + "/storage/metadata.pkl"

# Initialize index and metadata
create_new_index = False

# Check if index file exists
if os.path.exists(INDEX_PATH):
    try:
        index = faiss.read_index(INDEX_PATH)
        print(f"Loaded FAISS index with dimension: {index.d}")
    except Exception as e:
        print(f"Error loading index: {e}")
        create_new_index = True
else:
    print(f"Index file not found: {INDEX_PATH}")
    create_new_index = True

# Initialize metadata
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
else:
    metadata = {}

# Hash
def hash(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')

def persist():
    # Save metadata
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved index to {INDEX_PATH}")

# Insert new embeddings
def insert_embedding(text, embeddings, uuid):
    global index, create_new_index
    
    print(f"Text: {text}")
    print(f"UUID: {uuid}")

    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Reshape for storage
    embeddings = embeddings.astype(np.float32).reshape(1, -1)
    
    embedding_dim = embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create new index if needed
    if create_new_index:
        print(f"Creating new FAISS index with dimension {embedding_dim}")
        flat_index = faiss.IndexFlatL2(embedding_dim)  # Base index for L2 distance
        index = faiss.IndexIDMap(flat_index)  # Wrap it with IndexIDMap to support custom IDs
        create_new_index = False
        
    # Check if dimension matches the index
    if embedding_dim != index.d:
        print(f"WARNING: Embedding dimension ({embedding_dim}) doesn't match index dimension ({index.d})")
        print("Creating new index with the correct dimension")
        flat_index = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIDMap(flat_index)
    
    # Store embeddings in a persistent index using FAISS
    uuid_int = hash(uuid)
    print(f"Adding vector with ID: {uuid_int}")
    
    try:
        index.add_with_ids(embeddings, np.array([uuid_int], dtype=np.int64))
        print(f"Successfully added vector to index")
        
        # Store mapping
        metadata[uuid_int] = text
        
        # Persist
        persist()
        
        print("Stored!")
    except Exception as e:
        print(f"Error adding to index: {e}")

def find_similar_vectors(query_embedding, top_k=5):
    global index
    
    # Check if index exists
    if not 'index' in globals() or index is None:
        print("Error: No index exists yet")
        return []
    
    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.detach().cpu().numpy()

    # Reshape for search
    query_vector = query_embedding.astype(np.float32).reshape(1, -1)
    
    try:
        # Search the index
        distances, ids = index.search(query_vector, top_k)
        
        # Format results
        results = []
        for i, (dist, id_val) in enumerate(zip(distances[0], ids[0])):
            if id_val >= 0 and id_val in metadata:  # -1 means no result found
                results.append({
                    'id': int(id_val),
                    'text': metadata[id_val],
                    'distance': float(dist)
                })
        
        return results
    except Exception as e:
        print(f"Error searching index: {e}")
        return []
