package engine

import "go-inference/pkg/quant"

// KVCacheType defines the storage precision for key and value vectors.
type KVCacheType string

const (
	KVTypeF32  KVCacheType = "f32"
	KVTypeQ8_0 KVCacheType = "q8_0"
	KVTypeQ4_0 KVCacheType = "q4_0"
)

// KVCache holds the key and value states for all layers across sequence positions.
type KVCache struct {
	Type     KVCacheType
	Key      [][]float32
	Value    [][]float32
	KeyQ8    [][]byte
	ValueQ8  [][]byte
	KeyQ4    [][]byte
	ValueQ4  [][]byte
	MaxSeq   int
	KVDim    int
	CurPos   int
}

// NewKVCache allocates key and value buffers for layers up to maxSeq tokens with default F32 precision.
func NewKVCache(numLayers, maxSeq, kvDim int) *KVCache {
	return NewQuantizedKVCache(numLayers, maxSeq, kvDim, KVTypeF32)
}

// NewQuantizedKVCache allocates key and value buffers for layers with the specified precision type (f32, q8_0, q4_0).
func NewQuantizedKVCache(numLayers, maxSeq, kvDim int, kvType KVCacheType) *KVCache {
	cache := &KVCache{
		Type:   kvType,
		MaxSeq: maxSeq,
		KVDim:  kvDim,
		CurPos: 0,
	}

	switch kvType {
	case KVTypeQ8_0:
		bytesPerVec := ((kvDim + 31) / 32) * 34
		cache.KeyQ8 = make([][]byte, numLayers)
		cache.ValueQ8 = make([][]byte, numLayers)
		for i := 0; i < numLayers; i++ {
			cache.KeyQ8[i] = make([]byte, maxSeq*bytesPerVec)
			cache.ValueQ8[i] = make([]byte, maxSeq*bytesPerVec)
		}
	case KVTypeQ4_0:
		bytesPerVec := ((kvDim + 31) / 32) * 18
		cache.KeyQ4 = make([][]byte, numLayers)
		cache.ValueQ4 = make([][]byte, numLayers)
		for i := 0; i < numLayers; i++ {
			cache.KeyQ4[i] = make([]byte, maxSeq*bytesPerVec)
			cache.ValueQ4[i] = make([]byte, maxSeq*bytesPerVec)
		}
	default:
		cache.Type = KVTypeF32
		cache.Key = make([][]float32, numLayers)
		cache.Value = make([][]float32, numLayers)
		for i := 0; i < numLayers; i++ {
			cache.Key[i] = make([]float32, maxSeq*kvDim)
			cache.Value[i] = make([]float32, maxSeq*kvDim)
		}
	}
	return cache
}

// Reset clears the sequence position count.
func (c *KVCache) Reset() {
	c.CurPos = 0
}

// Write copies a key and value vector into the specified layer and slot.
func (c *KVCache) Write(layer, slot int, k, v []float32) {
	if slot < 0 || slot >= c.MaxSeq {
		return
	}

	switch c.Type {
	case KVTypeQ8_0:
		if layer < 0 || layer >= len(c.KeyQ8) {
			return
		}
		bytesPerVec := ((c.KVDim + 31) / 32) * 34
		offset := slot * bytesPerVec
		qk := quant.QuantizeQ8_0(k)
		qv := quant.QuantizeQ8_0(v)
		copy(c.KeyQ8[layer][offset:offset+bytesPerVec], qk)
		copy(c.ValueQ8[layer][offset:offset+bytesPerVec], qv)
	case KVTypeQ4_0:
		if layer < 0 || layer >= len(c.KeyQ4) {
			return
		}
		bytesPerVec := ((c.KVDim + 31) / 32) * 18
		offset := slot * bytesPerVec
		qk := quant.QuantizeQ4_0(k)
		qv := quant.QuantizeQ4_0(v)
		copy(c.KeyQ4[layer][offset:offset+bytesPerVec], qk)
		copy(c.ValueQ4[layer][offset:offset+bytesPerVec], qv)
	default:
		if layer < 0 || layer >= len(c.Key) {
			return
		}
		offset := slot * c.KVDim
		copy(c.Key[layer][offset:offset+c.KVDim], k)
		copy(c.Value[layer][offset:offset+c.KVDim], v)
	}
}

// Get returns the key and value sub-slices for the specified layer and slot.
func (c *KVCache) Get(layer, slot int) ([]float32, []float32) {
	if slot < 0 || slot >= c.MaxSeq {
		return nil, nil
	}

	switch c.Type {
	case KVTypeQ8_0:
		if layer < 0 || layer >= len(c.KeyQ8) {
			return nil, nil
		}
		bytesPerVec := ((c.KVDim + 31) / 32) * 34
		offset := slot * bytesPerVec
		k := quant.DequantizeQ8_0(c.KeyQ8[layer][offset:offset+bytesPerVec], c.KVDim)
		v := quant.DequantizeQ8_0(c.ValueQ8[layer][offset:offset+bytesPerVec], c.KVDim)
		return k, v
	case KVTypeQ4_0:
		if layer < 0 || layer >= len(c.KeyQ4) {
			return nil, nil
		}
		bytesPerVec := ((c.KVDim + 31) / 32) * 18
		offset := slot * bytesPerVec
		k := quant.DequantizeQ4_0(c.KeyQ4[layer][offset:offset+bytesPerVec], c.KVDim)
		v := quant.DequantizeQ4_0(c.ValueQ4[layer][offset:offset+bytesPerVec], c.KVDim)
		return k, v
	default:
		if layer < 0 || layer >= len(c.Key) {
			return nil, nil
		}
		offset := slot * c.KVDim
		return c.Key[layer][offset : offset+c.KVDim], c.Value[layer][offset : offset+c.KVDim]
	}
}
