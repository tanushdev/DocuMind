// Package api provides HTTP handlers for the vector service.
package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/documind/vector-service/internal/index"
	"github.com/documind/vector-service/pkg/types"
)

// Handler holds the dependencies for HTTP handlers.
type Handler struct {
	bruteForce *index.BruteForceIndex
	hnsw       *index.HNSWIndex
	dimensions int
}

// NewHandler creates a new Handler with initialized indexes.
func NewHandler(dimensions int) *Handler {
	return &Handler{
		bruteForce: index.NewBruteForceIndex(dimensions),
		hnsw:       index.NewHNSWIndex(index.DefaultHNSWConfig(dimensions)),
		dimensions: dimensions,
	}
}

// HandleInsert handles POST /insert requests.
func (h *Handler) HandleInsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req types.InsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendJSON(w, http.StatusBadRequest, types.InsertResponse{
			Success: false,
			Message: "Invalid JSON: " + err.Error(),
		})
		return
	}

	// Validate embedding dimensions
	if len(req.Embedding) != h.dimensions {
		sendJSON(w, http.StatusBadRequest, types.InsertResponse{
			Success: false,
			Message: "Invalid embedding dimensions",
		})
		return
	}

	// Create vector
	v := types.Vector{
		ID:        req.ID,
		Embedding: req.Embedding,
		Metadata:  req.Metadata,
	}

	// Insert into both indexes
	if err := h.bruteForce.Insert(v); err != nil {
		sendJSON(w, http.StatusInternalServerError, types.InsertResponse{
			Success: false,
			Message: "Failed to insert into brute-force index: " + err.Error(),
		})
		return
	}

	if err := h.hnsw.Insert(v); err != nil {
		sendJSON(w, http.StatusInternalServerError, types.InsertResponse{
			Success: false,
			Message: "Failed to insert into HNSW index: " + err.Error(),
		})
		return
	}

	sendJSON(w, http.StatusOK, types.InsertResponse{
		Success: true,
		Message: "Vector inserted successfully",
	})
}

// HandleInsertBatch handles POST /insert/batch requests.
func (h *Handler) HandleInsertBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req types.InsertBatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendJSON(w, http.StatusBadRequest, types.InsertBatchResponse{
			Inserted: 0,
			Message:  "Invalid JSON: " + err.Error(),
		})
		return
	}

	// Validate embeddings
	for _, v := range req.Vectors {
		if len(v.Embedding) != h.dimensions {
			sendJSON(w, http.StatusBadRequest, types.InsertBatchResponse{
				Inserted: 0,
				Message:  "Invalid embedding dimensions for vector: " + v.ID,
			})
			return
		}
	}

	// Insert into both indexes
	countBF, err := h.bruteForce.InsertBatch(req.Vectors)
	if err != nil {
		sendJSON(w, http.StatusInternalServerError, types.InsertBatchResponse{
			Inserted: 0,
			Message:  "Failed to insert into brute-force index: " + err.Error(),
		})
		return
	}

	_, err = h.hnsw.InsertBatch(req.Vectors)
	if err != nil {
		sendJSON(w, http.StatusInternalServerError, types.InsertBatchResponse{
			Inserted: countBF,
			Message:  "Failed to insert into HNSW index: " + err.Error(),
		})
		return
	}

	sendJSON(w, http.StatusOK, types.InsertBatchResponse{
		Inserted: countBF,
		Message:  "Vectors inserted successfully",
	})
}

// HandleSearch handles POST /search requests.
func (h *Handler) HandleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req types.SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON: " + err.Error()})
		return
	}

	// Validate
	if len(req.Embedding) != h.dimensions {
		sendJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid embedding dimensions"})
		return
	}

	if req.TopK <= 0 {
		req.TopK = 10 // Default
	}

	if req.Algorithm == "" {
		req.Algorithm = "hnsw" // Default to HNSW
	}

	start := time.Now()
	var results []types.SearchResult

	switch req.Algorithm {
	case "bruteforce":
		results = h.bruteForce.Search(req.Embedding, req.TopK)
	case "bruteforce_concurrent":
		results = h.bruteForce.SearchConcurrent(req.Embedding, req.TopK, 4)
	case "hnsw":
		results = h.hnsw.Search(req.Embedding, req.TopK)
	default:
		sendJSON(w, http.StatusBadRequest, map[string]string{"error": "Unknown algorithm: " + req.Algorithm})
		return
	}

	latencyMs := float64(time.Since(start).Microseconds()) / 1000.0

	sendJSON(w, http.StatusOK, types.SearchResponse{
		Results: results,
		Latency: latencyMs,
	})
}

// HandleHealth handles GET /health requests.
func (h *Handler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	sendJSON(w, http.StatusOK, types.HealthResponse{
		Status:      "ok",
		VectorCount: h.bruteForce.Count(),
	})
}

// HandleStats handles GET /stats requests.
func (h *Handler) HandleStats(w http.ResponseWriter, r *http.Request) {
	sendJSON(w, http.StatusOK, types.StatsResponse{
		VectorCount: h.bruteForce.Count(),
		Dimensions:  h.dimensions,
		IndexType:   "hnsw+bruteforce",
	})
}

// sendJSON sends a JSON response with the given status code.
func sendJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
