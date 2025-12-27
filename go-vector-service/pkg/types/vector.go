// Package types defines the core data types used throughout the vector service.
package types

// Vector represents a single embedding with its associated metadata.
// This is the fundamental data structure for storage and search operations.
type Vector struct {
	ID        string   `json:"id"`
	Embedding []float32 `json:"embedding"`
	Metadata  Metadata `json:"metadata"`
}

// Metadata contains information about the source document and chunk.
type Metadata struct {
	DocumentID string `json:"document_id"`
	ChunkIndex int    `json:"chunk_index"`
	Text       string `json:"text"`
	PageNumber int    `json:"page_number,omitempty"`
}

// SearchResult represents a single result from a vector search operation.
type SearchResult struct {
	ID       string   `json:"id"`
	Score    float32  `json:"score"`
	Metadata Metadata `json:"metadata"`
}

// InsertRequest is the request body for inserting a single vector.
type InsertRequest struct {
	ID        string    `json:"id"`
	Embedding []float32 `json:"embedding"`
	Metadata  Metadata  `json:"metadata"`
}

// InsertBatchRequest is the request body for batch vector insertion.
type InsertBatchRequest struct {
	Vectors []Vector `json:"vectors"`
}

// SearchRequest is the request body for vector search.
type SearchRequest struct {
	Embedding []float32 `json:"embedding"`
	TopK      int       `json:"top_k"`
	Algorithm string    `json:"algorithm"` // "bruteforce" or "hnsw"
}

// SearchResponse is the response body for vector search.
type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Latency float64        `json:"latency_ms"`
}

// InsertResponse is the response body for insert operations.
type InsertResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
}

// InsertBatchResponse is the response body for batch insert operations.
type InsertBatchResponse struct {
	Inserted int    `json:"inserted"`
	Message  string `json:"message,omitempty"`
}

// HealthResponse is the response body for health check.
type HealthResponse struct {
	Status      string `json:"status"`
	VectorCount int    `json:"vector_count"`
}

// StatsResponse is the response body for stats endpoint.
type StatsResponse struct {
	VectorCount int    `json:"vector_count"`
	Dimensions  int    `json:"dimensions"`
	IndexType   string `json:"index_type"`
}
