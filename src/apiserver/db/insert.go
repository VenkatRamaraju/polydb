package apiserver

import (
	"apiserver/agrpc"
	"bpe"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// global channel of requests
var ChannelInsertRequests = make(chan *InsertRequest)

// global map of channels for responses
var MapChannelResponse = make(map[string](chan *InsertResponse))

// map for calculating insertions
var mapMerges map[string]interface{}

// structs for state maintenance
type InsertRequest struct {
	Text string `json:"text"`
	UUID string
}

type InsertResponse struct {
	ID     string `json:"id,omitempty"`
	Status string `json:"status"` // "ok" or "error"
	Error  string `json:"error,omitempty"`
}

// Initialize
func Initialize() error {
	// Convert to tokens
	baseDir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get working directory: %w", err)
	}

	// Read merges map from JSON file
	pdFile, err := os.Open(filepath.Join(baseDir, "..", "..", "artifacts", "merges.json"))
	if err != nil {
		return fmt.Errorf("failed to open merges file: %w", err)
	}
	defer pdFile.Close()

	// Decode the JSON data into the map
	decoder := json.NewDecoder(pdFile)
	if err = decoder.Decode(&mapMerges); err != nil {
		return fmt.Errorf("failed to decode merges map: %w", err)
	}

	return nil
}

// insert into database
func Insert(sText string, sUUID string) *InsertResponse {
	// Convert to tokens
	alTokens, err := bpe.Encode(mapMerges, sText)
	if err != nil {
		return &InsertResponse{Status: "error", Error: err.Error()}
	}

	// insert into embeddings storage (faiss)
	// Connect to embeddings service
	embClient, err := agrpc.NewClient()
	if err != nil {
		return &InsertResponse{Status: "error", Error: fmt.Sprintf("Failed to connect to embeddings service: %v", err)}
	}
	defer embClient.Close()

	// Generate embeddings using gRPC
	result, err := embClient.GenerateEmbeddings(alTokens)
	if err != nil {
		return &InsertResponse{Status: "error", Error: fmt.Sprintf("Failed to generate embeddings: %v", err)}
	}

	fmt.Println(result)

	return nil
}

// process requests
func LaunchHandler() {
	for job := range ChannelInsertRequests {
		// call function
		response := Insert(job.Text, job.UUID)

		// insert response into map where caller is expecting it
		MapChannelResponse[job.UUID] <- response
	}
}
