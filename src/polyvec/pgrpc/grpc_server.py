import grpc
import sys
import os
import torch
import numpy as np
import concurrent.futures
from concurrent import futures
import time

# Define base path
BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOCAL_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIRECTORY)
sys.path.append(LOCAL_DIRECTORY)

# Import the embeddings module and proto-generated code
from train.embeddings import generate_embeddings
sys.path.append(os.path.join(BASE_DIRECTORY, 'src', 'polyvec', 'proto'))

# Import the generated proto classes (after generating them)
import embeddings_pb2
import embeddings_pb2_grpc

class EmbeddingsServicer(embeddings_pb2_grpc.EmbeddingsServicer):
    def GenerateEmbeddings(self, request, context):
        try:
            # Convert token IDs to tensor and generate embeddings
            token_ids = list(request.token_ids)
            embeddings = generate_embeddings(token_ids)
            
            # Insert into index
            print(embeddings)

            # Return
            return True
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error generating embeddings: {str(e)}")
            return False

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embeddings_pb2_grpc.add_EmbeddingsServicer_to_server(
        EmbeddingsServicer(), server)
    
    # Use a Unix socket for communication
    socket_path = '/tmp/embeddings.sock'
    
    # Remove existing socket file if it exists
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    
    server.add_insecure_port(f'unix:{socket_path}')
    server.start()
    print(f"Embeddings gRPC server running at {socket_path}")
    
    # Keep the server running
    try:
        while True:
            time.sleep(86400) 
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()