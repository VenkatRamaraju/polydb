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

# Initialize index and metadata
index = faiss.read_index(INDEX_PATH)

# Initialize metadata
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)


# Hash
def hash(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


# Insert new embeddings
def insert_embedding(text, embeddings, uuid):
    print(f"Text: {text}")
    print(f"UUID: {uuid}")

    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Reshape for storage
    embeddings = embeddings.astype(np.float32).reshape(1, -1)

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

def persist():
    # Save metadata
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved index to {INDEX_PATH}")


test_text = "venkat ramaraju"
test_uuid = "1"
test_embeddings = generate_embeddings([1, 2, 3, 4, 5])
print(f"Generated embeddings shape: {test_embeddings.shape}")
insert_embedding(test_text, test_embeddings, test_uuid)
