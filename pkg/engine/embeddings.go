package engine

import (
	"fmt"
	"math"
)

// Embed computes a normalized dense embedding vector for the given input text.
// It runs the transformer forward pass and performs mean pooling over token activations,
// followed by L2-normalization (standard for semantic search and vector databases).
func (e *Engine) Embed(prompt string) ([]float32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	tokens := e.Tokenizer.Encode(prompt, true)
	if len(tokens) == 0 {
		return nil, 0, fmt.Errorf("prompt produced 0 tokens")
	}

	if len(tokens) > e.Config.SeqLen {
		tokens = tokens[:e.Config.SeqLen]
	}

	kv := e.NewKVCache()
	dim := e.Config.Dim
	accum := make([]float32, dim)

	// Forward through layers and accumulate activations
	for pos, tok := range tokens {
		e.Forward(tok, pos, kv)
		// e.Arena.XB contains the post-layer / normalized activation for this token
		for i := 0; i < dim; i++ {
			accum[i] += e.Arena.XB[i]
		}
	}

	// Mean pooling across all tokens
	n := float32(len(tokens))
	var normSq float32
	for i := 0; i < dim; i++ {
		accum[i] /= n
		normSq += accum[i] * accum[i]
	}

	// L2-normalization: vec = vec / ||vec||
	if normSq > 0 {
		invNorm := float32(1.0 / math.Sqrt(float64(normSq)))
		for i := 0; i < dim; i++ {
			accum[i] *= invNorm
		}
	}

	return accum, len(tokens), nil
}
