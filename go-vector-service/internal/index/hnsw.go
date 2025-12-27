// Package index provides HNSW (Hierarchical Navigable Small World) implementation.
// This is a graph-based approximate nearest neighbor (ANN) algorithm that provides
// O(log n) search complexity compared to O(n) for brute-force.
//
// Reference: "Efficient and robust approximate nearest neighbor search using
// Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2016)
package index

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"

	"github.com/documind/vector-service/pkg/types"
)

// HNSWConfig contains configuration parameters for the HNSW index.
type HNSWConfig struct {
	// M is the maximum number of connections per node per layer.
	// Higher M = better recall but more memory and slower construction.
	// Typical values: 12-48
	M int

	// MMax is the maximum number of connections at layer 0.
	// Usually set to 2*M.
	MMax int

	// EfConstruction is the beam width during index construction.
	// Higher = better quality but slower construction.
	// Typical values: 100-200
	EfConstruction int

	// EfSearch is the beam width during search.
	// Higher = better recall but slower search.
	// Typical values: 50-200
	EfSearch int

	// ML is the level generation factor.
	// New nodes are assigned level floor(-ln(uniform(0,1)) * mL).
	// Typically set to 1/ln(M).
	ML float64

	// Dimensions is the vector dimensionality.
	Dimensions int
}

// DefaultHNSWConfig returns sensible defaults for HNSW.
func DefaultHNSWConfig(dimensions int) HNSWConfig {
	m := 16
	return HNSWConfig{
		M:              m,
		MMax:           m * 2,
		EfConstruction: 200,
		EfSearch:       100,
		ML:             1.0 / math.Log(float64(m)),
		Dimensions:     dimensions,
	}
}

// HNSWNode represents a node in the HNSW graph.
type HNSWNode struct {
	ID         string
	Vector     []float32
	Metadata   types.Metadata
	Neighbors  [][]string // neighbors[layer] = list of neighbor IDs
	Layer      int        // Maximum layer this node appears in
}

// HNSWIndex implements the HNSW algorithm for approximate nearest neighbor search.
type HNSWIndex struct {
	nodes      map[string]*HNSWNode
	entryPoint string
	maxLevel   int
	config     HNSWConfig
	mu         sync.RWMutex
	rng        *rand.Rand
}

// NewHNSWIndex creates a new HNSW index with the given configuration.
func NewHNSWIndex(config HNSWConfig) *HNSWIndex {
	return &HNSWIndex{
		nodes:    make(map[string]*HNSWNode),
		maxLevel: -1,
		config:   config,
		rng:      rand.New(rand.NewSource(42)), // Deterministic for reproducibility
	}
}

// Insert adds a vector to the HNSW index.
// Time Complexity: O(log n) average case
func (idx *HNSWIndex) Insert(v types.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Generate random level for new node
	level := idx.randomLevel()

	// Create new node
	node := &HNSWNode{
		ID:        v.ID,
		Vector:    v.Embedding,
		Metadata:  v.Metadata,
		Neighbors: make([][]string, level+1),
		Layer:     level,
	}

	for i := range node.Neighbors {
		node.Neighbors[i] = make([]string, 0)
	}

	idx.nodes[v.ID] = node

	// If this is the first node, set as entry point
	if idx.entryPoint == "" {
		idx.entryPoint = v.ID
		idx.maxLevel = level
		return nil
	}

	// Find entry point for insertion
	currentNode := idx.nodes[idx.entryPoint]
	entryID := idx.entryPoint

	// Traverse from top level down to level+1 (greedy search)
	for l := idx.maxLevel; l > level; l-- {
		changed := true
		for changed {
			changed = false
			if l < len(currentNode.Neighbors) {
				for _, neighborID := range currentNode.Neighbors[l] {
					neighbor := idx.nodes[neighborID]
					if neighbor == nil {
						continue
					}
					if CosineDistance(v.Embedding, neighbor.Vector) < CosineDistance(v.Embedding, currentNode.Vector) {
						currentNode = neighbor
						entryID = neighborID
						changed = true
					}
				}
			}
		}
	}

	// For each layer from min(level, maxLevel) down to 0
	for l := min(level, idx.maxLevel); l >= 0; l-- {
		// Find ef_construction nearest neighbors at this layer
		neighbors := idx.searchLayer(v.Embedding, entryID, idx.config.EfConstruction, l)

		// Select M best neighbors
		mLayer := idx.config.M
		if l == 0 {
			mLayer = idx.config.MMax
		}

		selectedNeighbors := idx.selectNeighbors(v.Embedding, neighbors, mLayer)

		// Add bidirectional connections
		node.Neighbors[l] = make([]string, 0, len(selectedNeighbors))
		for _, n := range selectedNeighbors {
			node.Neighbors[l] = append(node.Neighbors[l], n.ID)

			// Add reverse connection
			neighbor := idx.nodes[n.ID]
			if neighbor != nil && l < len(neighbor.Neighbors) {
				neighbor.Neighbors[l] = append(neighbor.Neighbors[l], v.ID)

				// Prune if exceeding maximum connections
				if len(neighbor.Neighbors[l]) > mLayer {
					neighbor.Neighbors[l] = idx.pruneConnections(neighbor.Vector, neighbor.Neighbors[l], mLayer)
				}
			}
		}

		// Update entry for next layer
		if len(selectedNeighbors) > 0 {
			entryID = selectedNeighbors[0].ID
		}
	}

	// Update entry point if new node has higher level
	if level > idx.maxLevel {
		idx.entryPoint = v.ID
		idx.maxLevel = level
	}

	return nil
}

// InsertBatch adds multiple vectors efficiently.
func (idx *HNSWIndex) InsertBatch(vectors []types.Vector) (int, error) {
	for _, v := range vectors {
		if err := idx.Insert(v); err != nil {
			return 0, err
		}
	}
	return len(vectors), nil
}

// Search finds the top-k approximate nearest neighbors.
// Time Complexity: O(log n) average case
func (idx *HNSWIndex) Search(query []float32, topK int) []types.SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.nodes) == 0 || idx.entryPoint == "" {
		return []types.SearchResult{}
	}

	// Start from entry point
	currentNode := idx.nodes[idx.entryPoint]
	entryID := idx.entryPoint

	// Traverse from top level down to level 1 (greedy search)
	for l := idx.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			if l < len(currentNode.Neighbors) {
				for _, neighborID := range currentNode.Neighbors[l] {
					neighbor := idx.nodes[neighborID]
					if neighbor == nil {
						continue
					}
					if CosineDistance(query, neighbor.Vector) < CosineDistance(query, currentNode.Vector) {
						currentNode = neighbor
						entryID = neighborID
						changed = true
					}
				}
			}
		}
	}

	// Search at layer 0 with ef_search width
	candidates := idx.searchLayer(query, entryID, idx.config.EfSearch, 0)

	// Return top-k results
	results := make([]types.SearchResult, 0, topK)
	for i := 0; i < len(candidates) && i < topK; i++ {
		node := idx.nodes[candidates[i].ID]
		if node != nil {
			results = append(results, types.SearchResult{
				ID:       candidates[i].ID,
				Score:    CosineSimilarity(query, node.Vector), // Convert distance to similarity
				Metadata: node.Metadata,
			})
		}
	}

	return results
}

// searchLayer performs beam search at a specific layer.
func (idx *HNSWIndex) searchLayer(query []float32, entryID string, ef int, layer int) []distanceNode {
	visited := make(map[string]bool)
	visited[entryID] = true

	entryNode := idx.nodes[entryID]
	if entryNode == nil {
		return nil
	}

	// Candidates (min-heap by distance - closest first)
	candidates := &distanceHeap{}
	heap.Init(candidates)

	// Dynamic list (max-heap by distance - furthest first for eviction)
	results := &distanceHeap{}
	heap.Init(results)

	entryDist := CosineDistance(query, entryNode.Vector)
	heap.Push(candidates, distanceNode{ID: entryID, Distance: entryDist, isMaxHeap: false})
	heap.Push(results, distanceNode{ID: entryID, Distance: entryDist, isMaxHeap: true})

	for candidates.Len() > 0 {
		// Get closest candidate
		closest := heap.Pop(candidates).(distanceNode)

		// Get furthest in results
		if results.Len() > 0 {
			furthest := (*results)[0]
			if closest.Distance > furthest.Distance {
				break // All remaining candidates are further than our worst result
			}
		}

		// Explore neighbors
		node := idx.nodes[closest.ID]
		if node == nil || layer >= len(node.Neighbors) {
			continue
		}

		for _, neighborID := range node.Neighbors[layer] {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			neighbor := idx.nodes[neighborID]
			if neighbor == nil {
				continue
			}

			dist := CosineDistance(query, neighbor.Vector)

			if results.Len() < ef {
				heap.Push(candidates, distanceNode{ID: neighborID, Distance: dist, isMaxHeap: false})
				heap.Push(results, distanceNode{ID: neighborID, Distance: dist, isMaxHeap: true})
			} else if dist < (*results)[0].Distance {
				heap.Push(candidates, distanceNode{ID: neighborID, Distance: dist, isMaxHeap: false})
				heap.Pop(results)
				heap.Push(results, distanceNode{ID: neighborID, Distance: dist, isMaxHeap: true})
			}
		}
	}

	// Extract results sorted by distance
	output := make([]distanceNode, results.Len())
	for i := len(output) - 1; i >= 0; i-- {
		output[i] = heap.Pop(results).(distanceNode)
	}

	// Sort by distance (closest first)
	for i := 0; i < len(output)/2; i++ {
		output[i], output[len(output)-1-i] = output[len(output)-1-i], output[i]
	}

	return output
}

// selectNeighbors selects the best M neighbors from candidates.
// Uses simple selection (can be enhanced with heuristic selection).
func (idx *HNSWIndex) selectNeighbors(query []float32, candidates []distanceNode, m int) []distanceNode {
	if len(candidates) <= m {
		return candidates
	}
	return candidates[:m]
}

// pruneConnections removes connections to maintain M limit.
func (idx *HNSWIndex) pruneConnections(nodeVector []float32, neighbors []string, m int) []string {
	if len(neighbors) <= m {
		return neighbors
	}

	// Calculate distances and sort
	type neighborDist struct {
		id   string
		dist float32
	}

	dists := make([]neighborDist, 0, len(neighbors))
	for _, nid := range neighbors {
		node := idx.nodes[nid]
		if node != nil {
			dists = append(dists, neighborDist{
				id:   nid,
				dist: CosineDistance(nodeVector, node.Vector),
			})
		}
	}

	// Sort by distance
	for i := 0; i < len(dists)-1; i++ {
		for j := i + 1; j < len(dists); j++ {
			if dists[j].dist < dists[i].dist {
				dists[i], dists[j] = dists[j], dists[i]
			}
		}
	}

	// Keep closest m
	result := make([]string, 0, m)
	for i := 0; i < m && i < len(dists); i++ {
		result = append(result, dists[i].id)
	}

	return result
}

// randomLevel generates a random level for a new node.
// Level distribution follows: P(level = l) = (1/M)^l * (1 - 1/M)
func (idx *HNSWIndex) randomLevel() int {
	level := 0
	for idx.rng.Float64() < idx.config.ML && level < 16 {
		level++
	}
	return level
}

// Count returns the number of vectors in the index.
func (idx *HNSWIndex) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.nodes)
}

// Dimensions returns the dimensionality of vectors in the index.
func (idx *HNSWIndex) Dimensions() int {
	return idx.config.Dimensions
}

// distanceNode pairs an ID with its distance for heap operations.
type distanceNode struct {
	ID        string
	Distance  float32
	isMaxHeap bool // If true, larger distance = higher priority
}

// distanceHeap implements heap.Interface for distanceNode.
type distanceHeap []distanceNode

func (h distanceHeap) Len() int { return len(h) }

func (h distanceHeap) Less(i, j int) bool {
	if len(h) > 0 && h[0].isMaxHeap {
		return h[i].Distance > h[j].Distance // Max-heap
	}
	return h[i].Distance < h[j].Distance // Min-heap
}

func (h distanceHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *distanceHeap) Push(x interface{}) {
	*h = append(*h, x.(distanceNode))
}

func (h *distanceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// min returns the smaller of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
