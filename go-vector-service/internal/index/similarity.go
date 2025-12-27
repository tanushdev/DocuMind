// Package index provides distance and similarity functions for vector operations.
// These are implemented manually without using external libraries to demonstrate
// algorithmic understanding.
package index

import (
	"math"
)

// CosineSimilarity calculates the cosine similarity between two vectors.
// Returns a value between -1 and 1, where 1 means identical direction,
// 0 means orthogonal, and -1 means opposite direction.
//
// Formula: cos(θ) = (A · B) / (||A|| × ||B||)
//
// Time Complexity: O(n) where n is the vector dimension
// Space Complexity: O(1)
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct float32
	var normA float32
	var normB float32

	// Single pass through both vectors for efficiency
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	// Avoid division by zero
	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// CosineDistance calculates the cosine distance between two vectors.
// Returns a value between 0 and 2, where 0 means identical direction.
// This is useful because lower distance = more similar (like L2).
//
// Formula: distance = 1 - cosine_similarity
func CosineDistance(a, b []float32) float32 {
	return 1 - CosineSimilarity(a, b)
}

// L2Distance calculates the Euclidean (L2) distance between two vectors.
// Returns a non-negative value, where 0 means identical vectors.
//
// Formula: ||A - B||₂ = √(Σ(aᵢ - bᵢ)²)
//
// Time Complexity: O(n) where n is the vector dimension
// Space Complexity: O(1)
func L2Distance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(math.MaxFloat32)
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return float32(math.Sqrt(float64(sum)))
}

// L2DistanceSquared calculates the squared Euclidean distance.
// This is faster than L2Distance as it avoids the square root.
// Use this when you only need to compare distances (ranking).
func L2DistanceSquared(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(math.MaxFloat32)
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sum
}

// DotProduct calculates the dot product of two vectors.
// For normalized vectors, this equals cosine similarity.
//
// Formula: A · B = Σ(aᵢ × bᵢ)
func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// Normalize normalizes a vector to unit length (L2 norm = 1).
// This is useful for converting to a form where dot product = cosine similarity.
func Normalize(v []float32) []float32 {
	if len(v) == 0 {
		return v
	}

	var norm float32
	for _, val := range v {
		norm += val * val
	}

	if norm == 0 {
		return v
	}

	norm = float32(math.Sqrt(float64(norm)))
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = val / norm
	}

	return result
}
