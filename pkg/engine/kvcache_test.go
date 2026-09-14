package engine

import (
	"testing"
)

func TestKVCacheCloneAndFork(t *testing.T) {
	types := []KVCacheType{KVTypeF32, KVTypeQ8_0, KVTypeQ4_0}
	numLayers := 2
	maxSeq := 16
	kvDim := 64

	for _, kvType := range types {
		t.Run(string(kvType), func(t *testing.T) {
			cache := NewQuantizedKVCache(numLayers, maxSeq, kvDim, kvType)
			cache.CurPos = 4

			// Populate slots 0, 1, 2
			kVec := make([]float32, kvDim)
			vVec := make([]float32, kvDim)
			for i := 0; i < kvDim; i++ {
				kVec[i] = float32(i + 1)
				vVec[i] = float32((i + 1) * 2)
			}
			cache.Write(0, 0, kVec, vVec)
			cache.Write(1, 1, kVec, vVec)

			// Clone cache
			clone := cache.Clone()
			if clone == nil {
				t.Fatalf("Clone returned nil")
			}
			if clone.CurPos != cache.CurPos {
				t.Errorf("CurPos mismatch: got %d, want %d", clone.CurPos, cache.CurPos)
			}
			if clone.Type != cache.Type {
				t.Errorf("Type mismatch: got %v, want %v", clone.Type, cache.Type)
			}

			// Verify cloned values
			gotK, gotV := clone.Get(0, 0)
			if len(gotK) != kvDim || len(gotV) != kvDim {
				t.Fatalf("Cloned Get returned unexpected slice lengths")
			}
			if kvType == KVTypeF32 {
				if gotK[0] != kVec[0] || gotV[0] != vVec[0] {
					t.Errorf("Cloned value mismatch on F32: got (%f, %f), want (%f, %f)", gotK[0], gotV[0], kVec[0], vVec[0])
				}
			}

			// ForkAt test
			forked := cache.ForkAt(1)
			if forked == nil {
				t.Fatalf("ForkAt returned nil")
			}
			gotForkK, _ := forked.Get(0, 0)
			if gotForkK == nil {
				t.Errorf("ForkAt slot 0 expected data, got nil")
			}

			// Verify mutation of clone does not affect original
			newK := make([]float32, kvDim)
			newK[0] = 999.0
			clone.Write(0, 0, newK, newK)
			origK, _ := cache.Get(0, 0)
			if kvType == KVTypeF32 && origK[0] == 999.0 {
				t.Errorf("Modifying clone affected original cache!")
			}
		})
	}
}
