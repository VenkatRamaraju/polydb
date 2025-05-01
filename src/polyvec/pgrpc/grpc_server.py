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
sys.path.append(os.path.join(BASE_DIRECTORY, 'src', 'polyvec', 'proto'))
from train.embeddings import generate_embeddings
from storage.insert import insert_embedding

# Import the generated proto classes (after generating them)
import embeddings_pb2
import embeddings_pb2_grpc

class EmbeddingsServicer(embeddings_pb2_grpc.EmbeddingsServicer):
    def GenerateEmbeddings(self, request, context):
        try:
            # Extract text and uuid from the metadata instead of from the request
            metadata = dict(context.invocation_metadata())
            text = metadata.get('text', 'unknown')
            uuid = metadata.get('uuid', 'unknown')
            
            # Get token IDs from the request
            token_ids = list(request.token_ids)
            
            # Generate embeddings
            embeddings = generate_embeddings(token_ids)

            # Insert into database and index
            insert_embedding(text, embeddings, uuid)

            # Create and return a proper response protobuf object
            response = embeddings_pb2.EmbeddingsResponse()
            response.success = True
            return response
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            response = embeddings_pb2.EmbeddingsResponse()
            response.success = False
            response.error_message = error_msg
            
            # Set gRPC status code for debugging but still return response object
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return response

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