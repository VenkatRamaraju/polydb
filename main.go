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
	"go.uber.org/zap"
)

func writeJSON(w http.ResponseWriter, status int, payload any) {
	// write to response body
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func makeInsertHandler(og *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// parse request
		var req apiserver.InsertRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, apiserver.InsertResponse{Status: "error", Error: "invalid JSON"})
			return
		}

		// insert function
		// if err := apiserver.Insert(req.Text); err != nil {
		// 	writeJSON(w, http.StatusBadRequest, insertResponse{Status: "error", Error: err.Error()})
		// 	return
		// }

		apiserver.ChannelInsertRequests <- req

		fmt.Println()

		// response
		writeJSON(w, http.StatusOK, apiserver.InsertResponse{Status: "ok"})
	}
}

// Orchestrate
func main() {
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

	// server
	srv := &http.Server{
		Addr:         ":9000",
		Handler:      r,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
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
