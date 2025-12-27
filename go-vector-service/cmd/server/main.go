// DocuMind Vector Service
// A high-performance vector search service built in Go.
//
// This service provides:
// - Brute-force exact nearest neighbor search
// - HNSW approximate nearest neighbor search
// - Concurrent search with goroutines
// - HTTP API for vector operations
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/documind/vector-service/internal/api"
)

func main() {
	// Command-line flags
	port := flag.Int("port", 8001, "Port to listen on")
	dimensions := flag.Int("dimensions", 384, "Vector dimensions (default: 384 for all-MiniLM-L6-v2)")
	flag.Parse()

	// Check for environment variable override
	if envPort := os.Getenv("VECTOR_SERVICE_PORT"); envPort != "" {
		fmt.Sscanf(envPort, "%d", port)
	}
	if envDim := os.Getenv("VECTOR_DIMENSIONS"); envDim != "" {
		fmt.Sscanf(envDim, "%d", dimensions)
	}

	// Initialize handler with indexes
	handler := api.NewHandler(*dimensions)

	// Create router
	router := api.NewRouter(handler)

	// Start server
	addr := fmt.Sprintf(":%d", *port)
	log.Printf("🚀 DocuMind Vector Service starting on %s", addr)
	log.Printf("📊 Vector dimensions: %d", *dimensions)
	log.Printf("📡 Endpoints:")
	log.Printf("   POST /insert       - Insert single vector")
	log.Printf("   POST /insert/batch - Insert multiple vectors")
	log.Printf("   POST /search       - Search for similar vectors")
	log.Printf("   GET  /health       - Health check")
	log.Printf("   GET  /stats        - Index statistics")

	if err := http.ListenAndServe(addr, router); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
