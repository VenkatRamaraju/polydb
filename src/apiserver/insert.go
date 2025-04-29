package apiserver

import "fmt"

// global channel of requests
ChannelInsertRequests := make(chan InsertRequest)

type InsertRequest struct {
	Text string `json:"text"`
}

type InsertResponse struct {
	ID     string `json:"id,omitempty"`
	Status string `json:"status"` // "ok" or "error"
	Error  string `json:"error,omitempty"`
}

func Insert() error {
	// generate token from strings
	fmt.Println()

	return nil
}
