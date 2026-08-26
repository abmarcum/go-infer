package engine

import (
	"sync"
)

// PrefixCache stores reusable KV-cache states indexed by token prefixes.
type PrefixCache struct {
	mu      sync.RWMutex
	entries map[string]*KVCacheEntry
	maxSize int
}

// KVCacheEntry holds a cached KV-cache state and its associated token sequence.
type KVCacheEntry struct {
	Tokens []int
	KV     *KVCache
}

// NewPrefixCache creates a thread-safe prefix cache with a maximum entry capacity.
func NewPrefixCache(maxSize int) *PrefixCache {
	if maxSize <= 0 {
		maxSize = 32
	}
	return &PrefixCache{
		entries: make(map[string]*KVCacheEntry),
		maxSize: maxSize,
	}
}

// tokensKey serializes a token slice into a string lookup key.
func tokensKey(tokens []int) string {
	var sb stringsBuilder
	for i, t := range tokens {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteInt(t)
	}
	return sb.String()
}

type stringsBuilder struct {
	buf []byte
}

func (sb *stringsBuilder) WriteByte(b byte) error {
	sb.buf = append(sb.buf, b)
	return nil
}

func (sb *stringsBuilder) WriteInt(n int) {
	if n == 0 {
		sb.buf = append(sb.buf, '0')
		return
	}
	if n < 0 {
		sb.buf = append(sb.buf, '-')
		n = -n
	}
	var digits [20]byte
	idx := len(digits)
	for n > 0 {
		idx--
		digits[idx] = byte('0' + (n % 10))
		n /= 10
	}
	sb.buf = append(sb.buf, digits[idx:]...)
}

func (sb *stringsBuilder) String() string {
	return string(sb.buf)
}

// FindLongestPrefix searches the cache for the longest matching token prefix.
// Returns the matched length, the cached KVCache copy, or 0 if no match exists.
func (pc *PrefixCache) FindLongestPrefix(tokens []int) (int, *KVCache) {
	if pc == nil || len(tokens) == 0 {
		return 0, nil
	}

	pc.mu.RLock()
	defer pc.mu.RUnlock()

	for l := len(tokens) - 1; l >= 1; l-- {
		key := tokensKey(tokens[:l])
		if entry, exists := pc.entries[key]; exists {
			// Deep copy the cached KV state up to prefix length
			clonedKV := cloneKVCachePrefix(entry.KV, l)
			return l, clonedKV
		}
	}

	return 0, nil
}

// Store inserts a completed prompt's KV-cache into the prefix cache.
func (pc *PrefixCache) Store(tokens []int, kv *KVCache) {
	if pc == nil || len(tokens) < 4 || kv == nil {
		return
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Simple eviction if exceeding capacity
	if len(pc.entries) >= pc.maxSize {
		for k := range pc.entries {
			delete(pc.entries, k)
			break
		}
	}

	key := tokensKey(tokens)
	pc.entries[key] = &KVCacheEntry{
		Tokens: append([]int{}, tokens...),
		KV:     cloneKVCachePrefix(kv, len(tokens)),
	}
}

// cloneKVCachePrefix creates a deep copy of a KV cache up to prefixLen slots.
func cloneKVCachePrefix(src *KVCache, prefixLen int) *KVCache {
	if src == nil {
		return nil
	}
	numLayers := len(src.Key)
	if numLayers == 0 {
		numLayers = len(src.KeyQ8)
	}
	if numLayers == 0 {
		numLayers = len(src.KeyQ4)
	}
	dst := NewQuantizedKVCache(numLayers, src.MaxSeq, src.KVDim, src.Type)

	for l := 0; l < numLayers; l++ {
		if len(src.Key) > l {
			copy(dst.Key[l], src.Key[l])
			copy(dst.Value[l], src.Value[l])
		}
		if len(src.KeyQ8) > l {
			copy(dst.KeyQ8[l], src.KeyQ8[l])
			copy(dst.ValueQ8[l], src.ValueQ8[l])
		}
		if len(src.KeyQ4) > l {
			copy(dst.KeyQ4[l], src.KeyQ4[l])
			copy(dst.ValueQ4[l], src.ValueQ4[l])
		}
	}
	return dst
}
