package apiserver

import (
	"agrpc"
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
var MapMerges map[string]interface{}

// embedding client
var embClient *agrpc.Client

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
	pdFile, err := os.Open(filepath.Join(baseDir, "artifacts", "merges.json"))
	if err != nil {
		return fmt.Errorf("failed to open merges file: %w", err)
	}
	defer pdFile.Close()

	// Decode the JSON data into the map
	decoder := json.NewDecoder(pdFile)
	if err = decoder.Decode(&MapMerges); err != nil {
		return fmt.Errorf("failed to decode merges map: %w", err)
	}

	// Connect to embeddings service
	embClient, err = agrpc.NewClient()
	if err != nil {
		return fmt.Errorf("failed to initialize embedding client: %w", err)
	}

	return nil
}

// insert into database
func Insert(sText string, sUUID string) *InsertResponse {
	// Convert to tokens
	alTokens, err := bpe.Encode(MapMerges, sText)
	if err != nil {
		return &InsertResponse{Status: "error", Error: err.Error()}
	}

	// Generate embeddings using gRPC
	err = embClient.GenerateEmbeddings(sText, alTokens, sUUID)
	if err != nil {
		return &InsertResponse{Status: "error", Error: fmt.Sprintf("Failed to generate embeddings: %v", err)}
	}

	return &InsertResponse{Status: "ok"}
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
