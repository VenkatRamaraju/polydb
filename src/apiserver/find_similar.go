package apiserver

import (
	"agrpc"
	"bpe"
	b64 "encoding/base64"
	"fmt"
)

// global channel of requests
var ChannelFindSimilarRequests = make(chan *FindSimilarRequest)

// global map of channels for responses
var MapChannelFindSimilarResponse = make(map[string](chan *FindSimilarResponse))

// structs for state maintenance
type FindSimilarRequest struct {
	Text string `json:"text"`
	TopK int32  `json:"top_k,omitempty"` // Optional: defaults to 5 if not specified
	UUID string
}

type FindSimilarResponse struct {
	SimilarTexts []string `json:"similar_texts"`
	Status       string   `json:"status"` // "ok" or "error"
	Error        string   `json:"error,omitempty"`
}

// FindSimilar finds similar texts to the provided text by using the embedding service
func FindSimilar(sText string, topK int32, sUUID string) *FindSimilarResponse {
	// Default value for topK if not provided
	if topK <= 0 {
		topK = 5
	}

	// Convert to tokens using the BPE encoder
	// This uses the same MapMerges variable from the insert.go file
	alTokens, err := bpe.Encode(MapMerges, sText)
	if err != nil {
		return &FindSimilarResponse{Status: "error", Error: err.Error()}
	}

	// Connect to embeddings service
	embClient, err := agrpc.NewClient()
	if err != nil {
		return &FindSimilarResponse{Status: "error", Error: fmt.Sprintf("Failed to connect to embeddings service: %v", err)}
	}
	defer embClient.Close()

	// Find similar embeddings using gRPC
	similarTexts, err := embClient.FindSimilarEmbeddings(alTokens, topK)
	if err != nil {
		return &FindSimilarResponse{Status: "error", Error: fmt.Sprintf("Failed to find similar embeddings: %v", err)}
	}

	// base 64 decode every string
	var asSimilarTextsDecoded []string
	for _, sEncodedText := range similarTexts {
		abDecoded, err := b64.StdEncoding.DecodeString(sEncodedText)
		if err != nil {
			return &FindSimilarResponse{Status: "error", Error: fmt.Sprintf("Unable to decode string: %v", err)}
		}
		asSimilarTextsDecoded = append(asSimilarTextsDecoded, string(abDecoded))
	}

	return &FindSimilarResponse{
		SimilarTexts: asSimilarTextsDecoded,
		Status:       "ok",
	}
}

// process find similar requests
func LaunchFindSimilarHandler() {
	for job := range ChannelFindSimilarRequests {
		// call function
		response := FindSimilar(job.Text, job.TopK, job.UUID)

		// insert response into map where caller is expecting it
		MapChannelFindSimilarResponse[job.UUID] <- response
	}
}
