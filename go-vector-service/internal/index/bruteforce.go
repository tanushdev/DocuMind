// Package index provides vector indexing implementations for similarity search.
package index

import (
	"container/heap"
	"sync"

	"github.com/documind/vector-service/pkg/types"
)

// BruteForceIndex implements exact nearest neighbor search using linear scan.
// This is the baseline implementation with O(n) search complexity.
// Use this for correctness verification and small datasets.
type BruteForceIndex struct {
	vectors    []types.Vector
	dimensions int
	mu         sync.RWMutex
}

// NewBruteForceIndex creates a new brute-force index.
func NewBruteForceIndex(dimensions int) *BruteForceIndex {
	return &BruteForceIndex{
		vectors:    make([]types.Vector, 0),
		dimensions: dimensions,
	}
}

// Insert adds a vector to the index.
// Thread-safe with write lock.
func (idx *BruteForceIndex) Insert(v types.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.vectors = append(idx.vectors, v)
	return nil
}

// InsertBatch adds multiple vectors to the index efficiently.
func (idx *BruteForceIndex) InsertBatch(vectors []types.Vector) (int, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.vectors = append(idx.vectors, vectors...)
	return len(vectors), nil
}

// Search finds the top-k most similar vectors using cosine similarity.
// Uses a min-heap to efficiently track top-k results.
// Time Complexity: O(n * d + n * log(k)) where n=vectors, d=dimensions, k=topK
func (idx *BruteForceIndex) Search(query []float32, topK int) []types.SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.vectors) == 0 {
		return []types.SearchResult{}
	}

	// Use a min-heap to track top-k results
	// We use a min-heap so we can efficiently remove the smallest score
	// when we find a better candidate
	h := &resultHeap{}
	heap.Init(h)

	for _, v := range idx.vectors {
		score := CosineSimilarity(query, v.Embedding)

		if h.Len() < topK {
			heap.Push(h, types.SearchResult{
				ID:       v.ID,
				Score:    score,
				Metadata: v.Metadata,
			})
		} else if score > (*h)[0].Score {
			// Replace the smallest score with this better one
			heap.Pop(h)
			heap.Push(h, types.SearchResult{
				ID:       v.ID,
				Score:    score,
				Metadata: v.Metadata,
			})
		}
	}

	// Extract results from heap and reverse to get descending order
	results := make([]types.SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(types.SearchResult)
	}

	return results
}

// SearchConcurrent performs parallel search using goroutines.
// Splits the vector space into chunks and searches in parallel.
// This demonstrates Go's concurrency model.
func (idx *BruteForceIndex) SearchConcurrent(query []float32, topK int, numWorkers int) []types.SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.vectors) == 0 {
		return []types.SearchResult{}
	}

	if numWorkers <= 0 {
		numWorkers = 4
	}

	// Calculate chunk size for each worker
	chunkSize := (len(idx.vectors) + numWorkers - 1) / numWorkers

	// Channel to collect results from workers
	resultsChan := make(chan []types.SearchResult, numWorkers)

	var wg sync.WaitGroup

	// Launch workers
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(idx.vectors) {
			end = len(idx.vectors)
		}
		if start >= len(idx.vectors) {
			break
		}

		wg.Add(1)
		go func(vectors []types.Vector) {
			defer wg.Done()
			results := searchChunk(query, vectors, topK)
			resultsChan <- results
		}(idx.vectors[start:end])
	}

	// Close channel when all workers complete
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Merge results from all workers
	return mergeResults(resultsChan, topK)
}

// searchChunk searches a subset of vectors and returns top-k results.
func searchChunk(query []float32, vectors []types.Vector, topK int) []types.SearchResult {
	h := &resultHeap{}
	heap.Init(h)

	for _, v := range vectors {
		score := CosineSimilarity(query, v.Embedding)

		if h.Len() < topK {
			heap.Push(h, types.SearchResult{
				ID:       v.ID,
				Score:    score,
				Metadata: v.Metadata,
			})
		} else if score > (*h)[0].Score {
			heap.Pop(h)
			heap.Push(h, types.SearchResult{
				ID:       v.ID,
				Score:    score,
				Metadata: v.Metadata,
			})
		}
	}

	results := make([]types.SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(types.SearchResult)
	}

	return results
}

// mergeResults combines results from multiple workers and returns top-k.
func mergeResults(resultsChan <-chan []types.SearchResult, topK int) []types.SearchResult {
	h := &resultHeap{}
	heap.Init(h)

	for partialResults := range resultsChan {
		for _, r := range partialResults {
			if h.Len() < topK {
				heap.Push(h, r)
			} else if r.Score > (*h)[0].Score {
				heap.Pop(h)
				heap.Push(h, r)
			}
		}
	}

	results := make([]types.SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(types.SearchResult)
	}

	return results
}

// Count returns the number of vectors in the index.
func (idx *BruteForceIndex) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// Dimensions returns the dimensionality of vectors in the index.
func (idx *BruteForceIndex) Dimensions() int {
	return idx.dimensions
}

// resultHeap is a min-heap for SearchResult, ordered by Score.
// We use a min-heap so we can efficiently evict the lowest score
// when we find a better candidate.
type resultHeap []types.SearchResult

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score } // Min-heap
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *resultHeap) Push(x interface{}) {
	*h = append(*h, x.(types.SearchResult))
}

func (h *resultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
