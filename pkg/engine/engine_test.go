package engine

import (
	"bytes"
	"encoding/binary"
	"go-inference/pkg/gguf"
	"go-inference/pkg/sampler"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// Helper to construct a minimal synthetic GGUF v3 file for testing
func createSyntheticGGUF(t *testing.T, dir string) string {
	dim := 16
	hiddenDim := 32
	numLayers := 1
	numHeads := 2
	numKVHeads := 2
	vocab := []string{"<unk>", "<s>", "</s>", "hello", "world", "go"}

	var buf bytes.Buffer

	// 1. Header
	binary.Write(&buf, binary.LittleEndian, uint32(gguf.Magic))
	binary.Write(&buf, binary.LittleEndian, uint32(gguf.Version3))

	// Tensor count:
	// token_embd.weight (vocab x dim)
	// blk.0.attn_norm.weight (dim)
	// blk.0.attn_q.weight (dim x dim)
	// blk.0.attn_k.weight (dim x dim)
	// blk.0.attn_v.weight (dim x dim)
	// blk.0.attn_output.weight (dim x dim)
	// blk.0.ffn_norm.weight (dim)
	// blk.0.ffn_gate.weight (hiddenDim x dim)
	// blk.0.ffn_up.weight (hiddenDim x dim)
	// blk.0.ffn_down.weight (dim x hiddenDim)
	// output_norm.weight (dim)
	// output.weight (vocab x dim)
	tensorNames := []string{
		"token_embd.weight",
		"blk.0.attn_norm.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"output_norm.weight",
		"output.weight",
	}
	tensorDims := [][]uint64{
		{uint64(dim), uint64(len(vocab))},
		{uint64(dim)},
		{uint64(dim), uint64(dim)},
		{uint64(dim), uint64(dim)},
		{uint64(dim), uint64(dim)},
		{uint64(dim), uint64(dim)},
		{uint64(dim)},
		{uint64(dim), uint64(hiddenDim)},
		{uint64(dim), uint64(hiddenDim)},
		{uint64(hiddenDim), uint64(dim)},
		{uint64(dim)},
		{uint64(dim), uint64(len(vocab))},
	}

	tensorCount := uint64(len(tensorNames))

	// Key-value metadata items:
	// llama.embedding_length (uint32)
	// llama.feed_forward_length (uint32)
	// llama.block_count (uint32)
	// llama.attention.head_count (uint32)
	// llama.attention.head_count_kv (uint32)
	// tokenizer.ggml.tokens (array of strings)
	// tokenizer.ggml.bos_token_id (uint32)
	// tokenizer.ggml.eos_token_id (uint32)
	kvCount := uint64(8)

	binary.Write(&buf, binary.LittleEndian, tensorCount)
	binary.Write(&buf, binary.LittleEndian, kvCount)

	// Write metadata helper
	writeStr := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	writeKVUint32 := func(k string, v uint32) {
		writeStr(k)
		binary.Write(&buf, binary.LittleEndian, uint32(gguf.TypeUint32))
		binary.Write(&buf, binary.LittleEndian, v)
	}

	writeKVUint32("llama.embedding_length", uint32(dim))
	writeKVUint32("llama.feed_forward_length", uint32(hiddenDim))
	writeKVUint32("llama.block_count", uint32(numLayers))
	writeKVUint32("llama.attention.head_count", uint32(numHeads))
	writeKVUint32("llama.attention.head_count_kv", uint32(numKVHeads))
	writeKVUint32("tokenizer.ggml.bos_token_id", 1)
	writeKVUint32("tokenizer.ggml.eos_token_id", 2)

	// Write tokens array
	writeStr("tokenizer.ggml.tokens")
	binary.Write(&buf, binary.LittleEndian, uint32(gguf.TypeArray))
	binary.Write(&buf, binary.LittleEndian, uint32(gguf.TypeString))
	binary.Write(&buf, binary.LittleEndian, uint64(len(vocab)))
	for _, tok := range vocab {
		writeStr(tok)
	}

	// Calculate tensor offsets and write tensor metadata
	var tensorOffset uint64
	for i, name := range tensorNames {
		writeStr(name)
		dims := tensorDims[i]
		binary.Write(&buf, binary.LittleEndian, uint32(len(dims)))
		for _, d := range dims {
			binary.Write(&buf, binary.LittleEndian, d)
		}
		binary.Write(&buf, binary.LittleEndian, uint32(gguf.GGMLTypeF32))
		binary.Write(&buf, binary.LittleEndian, tensorOffset)

		nElem := 1
		for _, d := range dims {
			nElem *= int(d)
		}
		tensorOffset += uint64(nElem * 4)
	}

	// 32-byte alignment padding for tensor data
	curLen := buf.Len()
	padding := (32 - (curLen % 32)) % 32
	buf.Write(make([]byte, padding))

	// Write tensor data (all 0.1 for float32 data)
	for _, dims := range tensorDims {
		nElem := 1
		for _, d := range dims {
			nElem *= int(d)
		}
		for e := 0; e < nElem; e++ {
			binary.Write(&buf, binary.LittleEndian, float32(0.1))
		}
	}

	filePath := filepath.Join(dir, "synthetic_model.gguf")
	if err := os.WriteFile(filePath, buf.Bytes(), 0644); err != nil {
		t.Fatalf("failed to write synthetic model: %v", err)
	}

	return filePath
}

func TestSyntheticModelInference(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := createSyntheticGGUF(t, tmpDir)

	eng, err := LoadModel(modelPath, 2)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer eng.Close()

	if eng.Config.Dim != 16 {
		t.Errorf("Expected dim 16, got %d", eng.Config.Dim)
	}
	if eng.Config.VocabSize != 6 {
		t.Errorf("Expected vocab size 6, got %d", eng.Config.VocabSize)
	}

	params := sampler.Params{Temperature: 0.0}
	var generatedTokens []string

	stats, err := eng.Generate("hello", 3, params, func(token string) bool {
		generatedTokens = append(generatedTokens, token)
		return true
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if stats.GeneratedTokens == 0 {
		t.Errorf("Expected generated tokens, got 0")
	}
	t.Logf("Stats: prompt=%d tokens, generated=%d tokens (tokens/sec=%.2f)",
		stats.PromptTokens, stats.GeneratedTokens, stats.TokensPerSecond)
}

func TestKVCacheContextWrap(t *testing.T) {
	numLayers := 2
	maxSeq := 4
	kvDim := 8
	kv := NewKVCache(numLayers, maxSeq, kvDim)

	// Simulate writing 10 tokens into a cache of maxSeq = 4
	for pos := 0; pos < 10; pos++ {
		slot := pos % maxSeq
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		for i := range k {
			k[i] = float32(pos*10 + i)
			v[i] = float32(pos*100 + i)
		}
		kv.Write(0, slot, k, v)
	}

	// Verify that the 10th write landed in slot 9 % 4 = 1
	kRead, vRead := kv.Get(0, 1)
	if kRead[0] != 90.0 || vRead[0] != 900.0 {
		t.Errorf("KVCache rolling wrap failed: expected k=90.0, v=900.0, got k=%f, v=%f", kRead[0], vRead[0])
	}
}

func TestBatchGEMMPrefillEquivalence(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := createSyntheticGGUF(t, tmpDir)

	eng, err := LoadModel(modelPath, 1)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer eng.Close()

	tokens := []int{1, 3, 4}
	kvSeq := eng.NewKVCache()
	var seqLogits []float32
	for i, tok := range tokens {
		seqLogits = eng.Forward(tok, i, kvSeq)
	}

	kvBatch := eng.NewKVCache()
	batchLogits := eng.ForwardBatch(tokens, kvBatch)

	if len(seqLogits) != len(batchLogits) {
		t.Fatalf("Logits length mismatch: seq=%d, batch=%d", len(seqLogits), len(batchLogits))
	}

	for i := range seqLogits {
		diff := seqLogits[i] - batchLogits[i]
		if diff < -1e-3 || diff > 1e-3 {
			t.Errorf("Mismatch between sequential and batch logits at %d: seq=%f, batch=%f", i, seqLogits[i], batchLogits[i])
		}
	}
}

func TestQuantizedKVCacheQ8_0(t *testing.T) {
	numLayers := 1
	maxSeq := 4
	kvDim := 32
	kv := NewQuantizedKVCache(numLayers, maxSeq, kvDim, KVTypeQ8_0)

	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	for i := range k {
		k[i] = float32(i)
		v[i] = float32(i * 2)
	}

	kv.Write(0, 0, k, v)
	kRead, vRead := kv.Get(0, 0)
	if len(kRead) != kvDim || len(vRead) != kvDim {
		t.Fatalf("Expected %d elements, got k=%d, v=%d", kvDim, len(kRead), len(vRead))
	}

	if math.Abs(float64(kRead[10]-10.0)) > 0.5 {
		t.Errorf("Q8_0 KV cache mismatch at index 10: got %f, expected 10.0", kRead[10])
	}
}

func TestQuantizedKVCacheQ4_0(t *testing.T) {
	numLayers := 1
	maxSeq := 4
	kvDim := 32
	kv := NewQuantizedKVCache(numLayers, maxSeq, kvDim, KVTypeQ4_0)

	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	for i := range k {
		k[i] = float32(i)
		v[i] = float32(i * 2)
	}

	kv.Write(0, 0, k, v)
	kRead, vRead := kv.Get(0, 0)
	if len(kRead) != kvDim || len(vRead) != kvDim {
		t.Fatalf("Expected %d elements, got k=%d, v=%d", kvDim, len(kRead), len(vRead))
	}

	if math.Abs(float64(kRead[10]-10.0)) > 3.0 {
		t.Errorf("Q4_0 KV cache mismatch at index 10: got %f, expected 10.0", kRead[10])
	}
}

func TestEngineConcurrentGeneration(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := createSyntheticGGUF(t, tmpDir)

	eng, err := LoadModel(modelPath, 1)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer eng.Close()

	params := sampler.DefaultParams()
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_, err := eng.Generate("hello world", 4, params, nil)
			if err != nil {
				t.Errorf("Concurrent generate %d failed: %v", id, err)
			}
		}(i)
	}

	wg.Wait()
}

func TestEngineMaxTokensCeiling(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := createSyntheticGGUF(t, tmpDir)

	eng, err := LoadModel(modelPath, 1)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer eng.Close()

	params := sampler.DefaultParams()
	// Requesting 999999 tokens when SeqLen is small (512)
	stats, err := eng.Generate("hello", 999999, params, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if stats.GeneratedTokens > eng.Config.SeqLen {
		t.Errorf("Generated tokens %d exceeded model SeqLen %d", stats.GeneratedTokens, eng.Config.SeqLen)
	}
}
