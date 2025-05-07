// main.go
package main

import (
	"apiserver"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

// displayBanner prints the PolyDB ASCII art and description to the console
func displayBanner() {
	banner := `
 ██████╗   ██████╗  ██╗   ██╗   ██╗ ██████╗  ██████╗  
 ██╔══██╗ ██╔═══██╗ ██║   ╚██╗ ██╔╝ ██╔══██╗ ██╔══██╗ 
 ██████╔╝ ██║   ██║ ██║    ╚████╔╝  ██║  ██║ ██████╔╝ 
 ██╔═══╝  ██║   ██║ ██║     ╚██╔╝   ██║  ██║ ██╔══██╗ 
 ██║      ╚██████╔╝ ███████╗ ██║    ██████╔╝ ██████╔╝ 
 ╚═╝       ╚═════╝  ╚══════╝ ╚═╝    ╚═════╝  ╚═════╝  
                                                       
   Vector Database
 ================================================================

    • Built from scratch with Go and Python
    • Strong multilingual tokenizer with > 3.0 compression across 10 unique scripts
    • Embedding model trained from scratch via Skip-Gram with Negative Sampling
    • High-performance vector operations using FAISS
    • Cross-language communication via GPRC

    Version: 1.0.0
    Starting server...
`
	fmt.Println(banner)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	// write to response body
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func makeInsertHandler(log *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Log the request
		log.Info("Received request", zap.String("method", r.Method), zap.String("url", r.URL.String()))

		// parse request
		var req apiserver.InsertRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Error("Failed to parse request", zap.Error(err), zap.String("endpoint", "/insert"))
			writeJSON(w, http.StatusBadRequest, apiserver.InsertResponse{Status: "error", Error: "invalid JSON"})
			return
		}

		// Validate that text field is present
		if req.Text == "" {
			log.Error("Missing required field: text", zap.String("endpoint", "/insert"))
			writeJSON(w, http.StatusBadRequest, apiserver.InsertResponse{Status: "error", Error: "missing required field: text"})
			return
		}

		// Log request details
		log.Info("Processing insert request", zap.String("text_length", fmt.Sprintf("%d chars", len(req.Text))))

		// insert request into channel
		req.UUID = uuid.New().String()

		// Create response channel first
		apiserver.MapChannelResponse[req.UUID] = make(chan *apiserver.InsertResponse, 1)
		apiserver.ChannelInsertRequests <- &req

		// block on response with timeout for response
		select {
		case res := <-apiserver.MapChannelResponse[req.UUID]:
			// response
			if res.Error == "" {
				log.Info("Insert request successful", zap.String("uuid", req.UUID))
				writeJSON(w, http.StatusOK, apiserver.InsertResponse{Status: res.Status})
			} else {
				log.Error("Insert request failed",
					zap.String("uuid", req.UUID),
					zap.String("error", res.Error))
				writeJSON(w, http.StatusOK, apiserver.InsertResponse{Status: res.Status, Error: res.Error})
			}
		case <-time.After(5 * time.Second):
			log.Error("Insert request timed out", zap.String("uuid", req.UUID))
			writeJSON(w, http.StatusOK, apiserver.InsertResponse{Error: "timeout"})
		}
		// Clean up the channel
		delete(apiserver.MapChannelFindSimilarResponse, req.UUID)
	}
}

func makeFindSimilarHandler(log *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Log the request
		log.Info("Received request", zap.String("method", r.Method), zap.String("url", r.URL.String()))

		// parse request
		var req apiserver.FindSimilarRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Error("Failed to parse request", zap.Error(err), zap.String("endpoint", "/find_similar"))
			writeJSON(w, http.StatusBadRequest, apiserver.FindSimilarResponse{Status: "error", Error: "invalid JSON"})
			return
		}

		// Validate that text field is present
		if req.Text == "" {
			log.Error("Missing required field: text", zap.String("endpoint", "/find_similar"))
			writeJSON(w, http.StatusBadRequest, apiserver.FindSimilarResponse{Status: "error", Error: "missing required field: text"})
			return
		}

		// Log request details
		log.Info("Processing find_similar request",
			zap.String("text_length", fmt.Sprintf("%d chars", len(req.Text))),
			zap.Int("top_k", int(req.TopK)))

		// Create response channel if it doesn't exist
		req.UUID = uuid.New().String()
		apiserver.MapChannelFindSimilarResponse[req.UUID] = make(chan *apiserver.FindSimilarResponse, 1)

		// Send request to handler
		apiserver.ChannelFindSimilarRequests <- &req

		// block on response with timeout for response
		select {
		case res := <-apiserver.MapChannelFindSimilarResponse[req.UUID]:
			// response
			if res.Error == "" {
				log.Info("Find similar request successful",
					zap.String("uuid", req.UUID),
					zap.Int("results_count", len(res.SimilarTexts)))
				writeJSON(w, http.StatusOK, apiserver.FindSimilarResponse{
					Status:       res.Status,
					SimilarTexts: res.SimilarTexts,
				})
			} else {
				log.Error("Find similar request failed",
					zap.String("uuid", req.UUID),
					zap.String("error", res.Error))
				writeJSON(w, http.StatusOK, apiserver.FindSimilarResponse{
					Status: res.Status,
					Error:  res.Error,
				})
			}
		case <-time.After(5 * time.Second):
			log.Error("Find similar request timed out", zap.String("uuid", req.UUID))
			writeJSON(w, http.StatusOK, apiserver.FindSimilarResponse{Error: "timeout"})
		}
		// Clean up the channel
		delete(apiserver.MapChannelFindSimilarResponse, req.UUID)
	}
}

// Orchestrate
func main() {
	// Display the PolyDB banner
	displayBanner()

	// logging
	log, _ := zap.NewProduction()
	defer log.Sync()

	// routing
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(10 * time.Second))

	r.Post("/insert", makeInsertHandler(log))
	r.Post("/find_similar", makeFindSimilarHandler(log))

	// Initialize API server
	log.Info("Initializing API server")
	err := apiserver.Initialize()
	if err != nil {
		log.Fatal("failed to initialize API server", zap.Error(err))
	}
	log.Info("API server initialized successfully")

	// server
	srv := &http.Server{
		Addr:         ":9000",
		Handler:      r,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// launch handler routines on apiserver
	HANDLERS := 100
	log.Info("Launching handler routines", zap.Int("count", HANDLERS))
	for iIndex := 0; iIndex < HANDLERS; iIndex++ {
		go apiserver.LaunchHandler()
		go apiserver.LaunchFindSimilarHandler()
	}

	// clean shutdown
	go func() {
		log.Info("API server listening", zap.String("addr", srv.Addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("listen", zap.Error(err))
		}
	}()

	// stop blocking when quit happens
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt)
	<-quit
	log.Info("shutting down server...")

	// clean everything up
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("server forced to shutdown", zap.Error(err))
	}
	log.Info("server exiting")
}
