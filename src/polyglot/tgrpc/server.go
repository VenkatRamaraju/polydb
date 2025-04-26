package tgrpc

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	pb "proto/tokenizerpb" // Local import reference to proto
)

type server struct {
	pb.UnimplementedTokenizerServer
	vocab map[string]int
}

// LoadVocab loads the vocabulary file
func LoadVocab() (map[string]interface{}, error) {
	// Define the base directory for the project
	baseDir, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("failed to get working directory: %w", err)
	}

	// Read merges map from JSON file
	pdFile, err := os.Open(filepath.Join(baseDir, "..", "..", "artifacts", "merges.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to open merges file: %w", err)
	}
	defer pdFile.Close()

	// Create a map to store the JSON data
	var artifactsMap map[string]interface{}

	// Decode the JSON data into the map
	decoder := json.NewDecoder(pdFile)
	if err = decoder.Decode(&artifactsMap); err != nil {
		return nil, fmt.Errorf("failed to decode merges map: %w", err)
	}

	return artifactsMap, nil
}

// Encode implements the Encode RPC method
func (s *server) Encode(ctx context.Context, req *pb.EncodeRequest) (*pb.EncodeResponse, error) {
	text := req.GetText()
	log.Printf("Received encode request: %s", text)

	// Simple tokenization (placeholder - implement your actual tokenization logic)
	tokens := []int64{}
	tokenTexts := []string{}

	// Split the text into words and encode using the vocabulary
	words := []string{text} // Simple example - in reality you'd split the text
	for _, word := range words {
		if id, exists := s.vocab[word]; exists {
			tokens = append(tokens, int64(id))
			tokenTexts = append(tokenTexts, word)
		}
	}

	return &pb.EncodeResponse{
		Tokens:     tokens,
		TokenTexts: tokenTexts,
	}, nil
}
