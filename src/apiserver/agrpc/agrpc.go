package agrpc

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "embeddingspb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// Client represents a gRPC client for the embeddings service
type Client struct {
	conn   *grpc.ClientConn
	client pb.EmbeddingsClient
}

// NewClient creates a new client connected to the embeddings gRPC service
func NewClient() (*Client, error) {
	// Connect to the Unix socket
	socketPath := "unix:///tmp/embeddings.sock"

	// Set up connection with retry logic
	var conn *grpc.ClientConn
	var err error

	for i := 0; i < 5; i++ {
		conn, err = grpc.Dial(
			socketPath,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithBlock(),
			grpc.WithTimeout(2*time.Second),
		)

		if err == nil {
			break
		}

		log.Printf("Failed to connect to embeddings service (attempt %d): %v", i+1, err)
		time.Sleep(1 * time.Second)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to connect to embeddings service after multiple attempts: %w", err)
	}

	client := pb.NewEmbeddingsClient(conn)
	return &Client{conn: conn, client: client}, nil
}

// GenerateEmbeddings sends token IDs to the embeddings service and returns the embedding vectors
func (c *Client) GenerateEmbeddings(tokenIDs []int64) ([]float32, int32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resp, err := c.client.GenerateEmbeddings(ctx, &pb.EmbeddingsRequest{
		TokenIds: tokenIDs,
	})

	if err != nil {
		if st, ok := status.FromError(err); ok {
			return nil, 0, fmt.Errorf("embeddings service error (%s): %s", st.Code(), st.Message())
		}
		return nil, 0, fmt.Errorf("failed to generate embeddings: %w", err)
	}

	return resp.EmbeddingVectors, resp.EmbeddingDim, nil
}

// Close closes the client connection
func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}
