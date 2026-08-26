package engine

import (
	"testing"
)

func TestPrefixCacheLookupAndStore(t *testing.T) {
	pc := NewPrefixCache(10)
	kv := NewKVCache(4, 128, 64)

	// Simulate populating KV cache with non-zero activations
	for l := 0; l < 4; l++ {
		for i := 0; i < 64*10; i++ {
			kv.Key[l][i] = float32(i + 1)
			kv.Value[l][i] = float32(i + 2)
		}
	}

	tokens := []int{101, 202, 303, 404, 505}
	pc.Store(tokens, kv)

	// Exact match search for a longer extension of tokens
	queryTokens := []int{101, 202, 303, 404, 505, 606, 707}
	matchedLen, clonedKV := pc.FindLongestPrefix(queryTokens)

	if matchedLen != 5 {
		t.Fatalf("Expected matched length 5, got %d", matchedLen)
	}
	if clonedKV == nil {
		t.Fatalf("Expected non-nil cloned KV cache")
	}

	// Verify cloned contents match
	if clonedKV.Key[0][0] != kv.Key[0][0] {
		t.Errorf("Cloned KV data mismatch: got %f, want %f", clonedKV.Key[0][0], kv.Key[0][0])
	}
}

func TestPrefixCacheMiss(t *testing.T) {
	pc := NewPrefixCache(10)
	queryTokens := []int{999, 888, 777}
	matchedLen, clonedKV := pc.FindLongestPrefix(queryTokens)

	if matchedLen != 0 || clonedKV != nil {
		t.Errorf("Expected miss (0, nil), got (%d, %v)", matchedLen, clonedKV)
	}
}
