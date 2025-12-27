// Package api provides HTTP routing for the vector service.
package api

import (
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"
)

// loggingMiddleware logs request details and latency.
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s - %v", r.Method, r.URL.Path, time.Since(start))
	})
}

// corsMiddleware adds CORS headers for development.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// NewRouter creates and configures the HTTP router.
func NewRouter(handler *Handler) *mux.Router {
	r := mux.NewRouter()

	// Apply middleware
	r.Use(loggingMiddleware)
	r.Use(corsMiddleware)

	// Register routes
	r.HandleFunc("/insert", handler.HandleInsert).Methods("POST", "OPTIONS")
	r.HandleFunc("/insert/batch", handler.HandleInsertBatch).Methods("POST", "OPTIONS")
	r.HandleFunc("/search", handler.HandleSearch).Methods("POST", "OPTIONS")
	r.HandleFunc("/health", handler.HandleHealth).Methods("GET")
	r.HandleFunc("/stats", handler.HandleStats).Methods("GET")

	return r
}
