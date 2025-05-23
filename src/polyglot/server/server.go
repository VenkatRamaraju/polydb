package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"

	"bpe"
)

// Global variable to store the merges map
var mapMerges map[string]interface{}
var pdSync sync.Once

// enableCORS sets the necessary headers for Cross-Origin Resource Sharing
func enableCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

// Launch the server
func Launch() {
	// Pre-load the merges map
	pdSync.Do(func() {
		var err error
		mapMerges, err = bpe.LoadMergesMap()
		if err != nil {
			log.Fatalf("Failed to load merges map: %s", err)
		}
	})

	// Set up handlers with CORS middleware
	http.HandleFunc("/encode", encodeHandler)
	http.HandleFunc("/decode", decodeHandler)
	http.HandleFunc("/vocabulary-size", vocabularySizeHandler)

	fmt.Println("Server starting on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Server failed: %s", err)
	}
}

// Response structure for the encode endpoint
type EncodeResponse struct {
	Tokens     []int64  `json:"tokens"`
	TokenTexts []string `json:"token_texts"`
}

// encodeHandler handles the /encode endpoint
func encodeHandler(dataWriter http.ResponseWriter, pdRequest *http.Request) {
	// Enable CORS for all requests
	enableCORS(dataWriter)

	// Handle preflight OPTIONS request
	if pdRequest.Method == http.MethodOptions {
		dataWriter.WriteHeader(http.StatusOK)
		return
	}

	// Only accept POST requests
	if pdRequest.Method != http.MethodPost {
		http.Error(dataWriter, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Retrieve the input string from the HTTP request
	var sInput string
	if err := json.NewDecoder(pdRequest.Body).Decode(&sInput); err != nil {
		http.Error(dataWriter, "Invalid input, expected a JSON string", http.StatusBadRequest)
		return
	}

	// get a part of it
	convertedMap, tfOK := mapMerges["merges"].(map[string]interface{})
	if !tfOK {
		http.Error(dataWriter, "Failed to convert map: merges key not found or not a map", http.StatusInternalServerError)
		return
	}

	// Call bpe.Encode() with the input string
	alEncodedTokens, err := bpe.Encode(convertedMap, sInput)
	if err != nil {
		http.Error(dataWriter, fmt.Sprintf("Encoding error: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert tokens to text representations
	asTokenTexts, err := bpe.ListToTokens(alEncodedTokens, convertedMap)
	if err != nil {
		http.Error(dataWriter, fmt.Sprintf("Token text conversion error: %v", err), http.StatusInternalServerError)
		return
	}

	// Create response
	dataResponse := EncodeResponse{
		Tokens:     alEncodedTokens,
		TokenTexts: asTokenTexts,
	}

	// Return the encoded result as the HTTP response
	dataWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(dataWriter).Encode(dataResponse); err != nil {
		http.Error(dataWriter, "Error encoding response", http.StatusInternalServerError)
		return
	}
}

// Request structure for the decode endpoint
type DecodeRequest struct {
	Tokens []int64 `json:"tokens"`
}

// decodeHandler handles the /decode endpoint
func decodeHandler(dataWriter http.ResponseWriter, pdRequest *http.Request) {
	// Enable CORS for all requests
	enableCORS(dataWriter)

	// Handle preflight OPTIONS request
	if pdRequest.Method == http.MethodOptions {
		dataWriter.WriteHeader(http.StatusOK)
		return
	}

	// Only accept POST requests
	if pdRequest.Method != http.MethodPost {
		http.Error(dataWriter, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Retrieve the input tokens from the HTTP request
	var request DecodeRequest
	if err := json.NewDecoder(pdRequest.Body).Decode(&request); err != nil {
		http.Error(dataWriter, "Invalid input, expected a JSON object with 'tokens' field", http.StatusBadRequest)
		return
	}

	// Call bpe.Decode() with the input tokens
	sDecodedString, err := bpe.Decode(mapMerges, request.Tokens)
	if err != nil {
		http.Error(dataWriter, fmt.Sprintf("Decoding error: %v", err), http.StatusInternalServerError)
		return
	}

	// Return the decoded string as the HTTP response
	dataWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(dataWriter).Encode(sDecodedString); err != nil {
		http.Error(dataWriter, "Error encoding response", http.StatusInternalServerError)
		return
	}
}

// vocabularySizeHandler handles the /vocabulary-size endpoint
func vocabularySizeHandler(dataWriter http.ResponseWriter, pdRequest *http.Request) {
	// Enable CORS for all requests
	enableCORS(dataWriter)

	// Handle preflight OPTIONS request
	if pdRequest.Method == http.MethodOptions {
		dataWriter.WriteHeader(http.StatusOK)
		return
	}

	// Only accept GET requests
	if pdRequest.Method != http.MethodGet {
		http.Error(dataWriter, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Call bpe.GetVocabularySize
	vocabSize, err := bpe.GetHigestToken()
	if err != nil {
		http.Error(dataWriter, fmt.Sprintf("Error getting vocabulary size: %v", err), http.StatusInternalServerError)
		return
	}

	// Return the vocabulary size as the HTTP response
	dataWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(dataWriter).Encode(map[string]int64{"vocabulary_size": vocabSize}); err != nil {
		http.Error(dataWriter, "Error encoding response", http.StatusInternalServerError)
		return
	}
}
