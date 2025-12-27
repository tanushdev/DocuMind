// Benchmark tests for the vector service.
// Run with: go test -bench=. -benchmem
package index_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/documind/vector-service/internal/index"
	"github.com/documind/vector-service/pkg/types"
)

const (
	dimensions = 384
	topK       = 10
)

// generateRandomVector creates a random vector of the given dimensions.
func generateRandomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}
	return v
}

// generateRandomVectors creates n random vectors.
func generateRandomVectors(n, dim int) []types.Vector {
	vectors := make([]types.Vector, n)
	for i := range vectors {
		vectors[i] = types.Vector{
			ID:        fmt.Sprintf("vec_%d", i),
			Embedding: generateRandomVector(dim),
			Metadata: types.Metadata{
				DocumentID: fmt.Sprintf("doc_%d", i/10),
				ChunkIndex: i % 10,
				Text:       fmt.Sprintf("Sample text chunk %d", i),
			},
		}
	}
	return vectors
}

// BenchmarkBruteForceSearch benchmarks brute-force search with varying dataset sizes.
func BenchmarkBruteForceSearch(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			// Setup
			idx := index.NewBruteForceIndex(dimensions)
			vectors := generateRandomVectors(size, dimensions)
			idx.InsertBatch(vectors)
			query := generateRandomVector(dimensions)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, topK)
			}
		})
	}
}

// BenchmarkBruteForceSearchConcurrent benchmarks concurrent brute-force search.
func BenchmarkBruteForceSearchConcurrent(b *testing.B) {
	sizes := []int{1000, 10000}
	workers := []int{2, 4, 8}

	for _, size := range sizes {
		for _, numWorkers := range workers {
			b.Run(fmt.Sprintf("n=%d/workers=%d", size, numWorkers), func(b *testing.B) {
				idx := index.NewBruteForceIndex(dimensions)
				vectors := generateRandomVectors(size, dimensions)
				idx.InsertBatch(vectors)
				query := generateRandomVector(dimensions)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idx.SearchConcurrent(query, topK, numWorkers)
				}
			})
		}
	}
}

// BenchmarkHNSWSearch benchmarks HNSW search with varying dataset sizes.
func BenchmarkHNSWSearch(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			config := index.DefaultHNSWConfig(dimensions)
			idx := index.NewHNSWIndex(config)
			vectors := generateRandomVectors(size, dimensions)

			// Build index (not timed)
			for _, v := range vectors {
				idx.Insert(v)
			}
			query := generateRandomVector(dimensions)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, topK)
			}
		})
	}
}

// BenchmarkHNSWInsert benchmarks HNSW insertion.
func BenchmarkHNSWInsert(b *testing.B) {
	config := index.DefaultHNSWConfig(dimensions)
	vectors := generateRandomVectors(b.N, dimensions)

	b.ResetTimer()
	idx := index.NewHNSWIndex(config)
	for i := 0; i < b.N; i++ {
		idx.Insert(vectors[i])
	}
}

// BenchmarkCosineSimilarity benchmarks the cosine similarity calculation.
func BenchmarkCosineSimilarity(b *testing.B) {
	a := generateRandomVector(dimensions)
	c := generateRandomVector(dimensions)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index.CosineSimilarity(a, c)
	}
}

// TestBruteForceCorrectness ensures brute-force returns correct results.
func TestBruteForceCorrectness(t *testing.T) {
	idx := index.NewBruteForceIndex(3)

	// Insert known vectors
	idx.Insert(types.Vector{ID: "a", Embedding: []float32{1, 0, 0}})
	idx.Insert(types.Vector{ID: "b", Embedding: []float32{0, 1, 0}})
	idx.Insert(types.Vector{ID: "c", Embedding: []float32{0.9, 0.1, 0}})

	// Query should return "a" as closest, then "c"
	results := idx.Search([]float32{1, 0, 0}, 2)

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	if results[0].ID != "a" {
		t.Errorf("Expected first result to be 'a', got '%s'", results[0].ID)
	}

	if results[1].ID != "c" {
		t.Errorf("Expected second result to be 'c', got '%s'", results[1].ID)
	}
}

// TestHNSWCorrectness ensures HNSW returns reasonable results.
func TestHNSWCorrectness(t *testing.T) {
	config := index.DefaultHNSWConfig(3)
	idx := index.NewHNSWIndex(config)

	// Insert known vectors
	idx.Insert(types.Vector{ID: "a", Embedding: []float32{1, 0, 0}})
	idx.Insert(types.Vector{ID: "b", Embedding: []float32{0, 1, 0}})
	idx.Insert(types.Vector{ID: "c", Embedding: []float32{0.9, 0.1, 0}})

	// Query should return "a" as closest
	results := idx.Search([]float32{1, 0, 0}, 2)

	if len(results) < 1 {
		t.Fatalf("Expected at least 1 result, got %d", len(results))
	}

	if results[0].ID != "a" {
		t.Errorf("Expected first result to be 'a', got '%s'", results[0].ID)
	}
}
