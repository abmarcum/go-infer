package engine

import (
	"fmt"
	"go-inference/pkg/gguf"
	"go-inference/pkg/math"
	"go-inference/pkg/metal"
	"go-inference/pkg/sampler"
	"go-inference/pkg/tokenizer"
	"strings"
	"sync"
	"time"
	"unsafe"
)

// Engine is the central inference engine orchestrator.
type Engine struct {
	Config             ModelConfig
	Reader             *gguf.Reader
	Tokenizer          *tokenizer.Tokenizer
	GEMV               *math.GEMVEngine
	Arena              *MemoryArena
	Weights            *Weights
	MetalLayers        []metal.LayerWeights
	PreallocatedLayers *metal.PreallocatedLayers
	OutNormBuf         unsafe.Pointer
	OutWeightBuf       unsafe.Pointer
	OutWeightTyp       int
	mu                 sync.Mutex
}

// ChatMessage represents a single message in a multi-turn chat.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// LoadModel opens and initializes a GGUF model file into the inference engine.
func LoadModel(filePath string, numThreads int) (*Engine, error) {
	reader, err := gguf.OpenFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}

	meta := reader.Header.Metadata

	// Parse vocabulary and merges for tokenizer
	var vocab, merges []string
	if rawTokens, ok := meta["tokenizer.ggml.tokens"].([]interface{}); ok {
		for _, t := range rawTokens {
			if s, ok := t.(string); ok {
				vocab = append(vocab, s)
			}
		}
	}
	if rawMerges, ok := meta["tokenizer.ggml.merges"].([]interface{}); ok {
		for _, m := range rawMerges {
			if s, ok := m.(string); ok {
				merges = append(merges, s)
			}
		}
	}

	bosID := int(gguf.GetMetadataUint(meta, "tokenizer.ggml.bos_token_id", 128000))
	eosID := int(gguf.GetMetadataUint(meta, "tokenizer.ggml.eos_token_id", 128001))

	arch := gguf.GetMetadataString(meta, "general.architecture", "llama")

	// Extract Hyperparameters dynamically based on architecture prefix
	getParamUint := func(suffix string, def uint64) uint64 {
		if v := gguf.GetMetadataUint(meta, fmt.Sprintf("%s.%s", arch, suffix), 0); v > 0 {
			return v
		}
		if v := gguf.GetMetadataUint(meta, fmt.Sprintf("llama.%s", suffix), 0); v > 0 {
			return v
		}
		return def
	}

	getParamFloat := func(suffix string, def float64) float64 {
		if v := gguf.GetMetadataFloat(meta, fmt.Sprintf("%s.%s", arch, suffix), 0); v > 0 {
			return v
		}
		if v := gguf.GetMetadataFloat(meta, fmt.Sprintf("llama.%s", suffix), 0); v > 0 {
			return v
		}
		return def
	}

	dim := int(getParamUint("embedding_length", 2048))
	hiddenDim := int(getParamUint("feed_forward_length", 5632))
	numLayers := int(getParamUint("block_count", 16))
	numHeads := int(getParamUint("attention.head_count", 32))
	numKVHeads := int(getParamUint("attention.head_count_kv", uint64(numHeads)))
	seqLen := int(getParamUint("context_length", 2048))
	ropeTheta := float32(getParamFloat("rope.freq_base", 500000.0))
	eps := float32(getParamFloat("attention.layer_norm_rms_epsilon", 1e-5))
	if eps == 0 {
		eps = float32(getParamFloat("attention.layer_norm_epsilon", 1e-5))
	}

	vocabSize := len(vocab)
	if vocabSize == 0 {
		// Fallback to output or embedding tensor dimensions
		if emb, ok := reader.Header.Tensors["token_embd.weight"]; ok && len(emb.Dimensions) > 1 {
			vocabSize = int(emb.Dimensions[1])
		}
	}

	tok := tokenizer.NewTokenizer(vocab, merges, bosID, eosID)

	cfg := ModelConfig{
		Dim:        dim,
		HiddenDim:  hiddenDim,
		NumLayers:  numLayers,
		NumHeads:   numHeads,
		NumKVHeads: numKVHeads,
		VocabSize:  vocabSize,
		SeqLen:     seqLen,
		RopeTheta:  ropeTheta,
		Eps:        eps,
		BosID:      bosID,
		EosID:      eosID,
		EotID:      tok.EotTokenID,
	}

	// Try initializing Apple Metal GPU on macOS
	_ = metal.Init()
	if metal.IsAvailable() {
		_ = metal.AllocBuffers(cfg.Dim, cfg.HiddenDim, cfg.KVDim(), cfg.VocabSize, cfg.NumLayers, cfg.SeqLen)
	}

	weights, err := NewWeights(reader)
	if err != nil {
		reader.Close()
		return nil, fmt.Errorf("load weights: %w", err)
	}

	arena := NewMemoryArena(cfg)
	gemv := math.NewGEMVEngine(numThreads)

	var metalLayers []metal.LayerWeights
	var outNormBuf, outWeightBuf unsafe.Pointer
	var outWeightTyp int

	if metal.IsAvailable() {
		metalLayers = make([]metal.LayerWeights, numLayers)
		for l := 0; l < numLayers; l++ {
			wqName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_q.weight", l))
			wkName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_k.weight", l))
			wvName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_v.weight", l))
			woName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_output.weight", l))
			gateName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_gate.weight", l))
			upName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_up.weight", l))
			downName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_down.weight", l))
			attnNormName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_norm.weight", l))
			ffnNormName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_norm.weight", l))

			metalLayers[l] = metal.LayerWeights{
				WQBuf:       weights.GPUBufs[wqName],
				WQType:      int(weights.Meta[wqName].Type),
				WKBuf:       weights.GPUBufs[wkName],
				WKType:      int(weights.Meta[wkName].Type),
				WVBuf:       weights.GPUBufs[wvName],
				WVType:      int(weights.Meta[wvName].Type),
				WOBuf:       weights.GPUBufs[woName],
				WOType:      int(weights.Meta[woName].Type),
				FFNGateBuf:  weights.GPUBufs[gateName],
				FFNGateType: int(weights.Meta[gateName].Type),
				FFNUpBuf:    weights.GPUBufs[upName],
				FFNUpType:   int(weights.Meta[upName].Type),
				FFNDownBuf:  weights.GPUBufs[downName],
				FFNDownType: int(weights.Meta[downName].Type),
				AttnNormBuf: weights.GPUBufs[attnNormName],
				FFNNormBuf:  weights.GPUBufs[ffnNormName],
			}
		}

		outNormName, _ := weights.ResolveTensorName("output_norm.weight")
		outWeightName, _ := weights.ResolveTensorName("output.weight")
		outNormBuf = weights.GPUBufs[outNormName]
		outWeightBuf = weights.GPUBufs[outWeightName]
		outWeightTyp = int(weights.Meta[outWeightName].Type)
	}

	preallocatedLayers := metal.NewPreallocatedLayers(metalLayers)

	return &Engine{
		Config:             cfg,
		Reader:             reader,
		Tokenizer:          tok,
		GEMV:               gemv,
		Arena:              arena,
		Weights:            weights,
		MetalLayers:        metalLayers,
		PreallocatedLayers: preallocatedLayers,
		OutNormBuf:         outNormBuf,
		OutWeightBuf:       outWeightBuf,
		OutWeightTyp:       outWeightTyp,
	}, nil
}

// Close frees model resources and memory mappings.
func (e *Engine) Close() error {
	if e.PreallocatedLayers != nil {
		e.PreallocatedLayers.Free()
	}
	if e.Weights != nil {
		e.Weights.Close()
	}
	if e.Reader != nil {
		return e.Reader.Close()
	}
	return nil
}

// NewKVCache allocates a KV cache suited for this engine's model with default F32 precision.
func (e *Engine) NewKVCache() *KVCache {
	return NewKVCache(e.Config.NumLayers, e.Config.SeqLen, e.Config.KVDim())
}

// NewQuantizedKVCache allocates a KV cache with the specified precision type (f32, q8_0, q4_0).
func (e *Engine) NewQuantizedKVCache(kvType KVCacheType) *KVCache {
	return NewQuantizedKVCache(e.Config.NumLayers, e.Config.SeqLen, e.Config.KVDim(), kvType)
}

// GenerateStats contains timing and token statistics.
type GenerateStats struct {
	PromptTokens     int
	GeneratedTokens  int
	PrefillDuration  time.Duration
	GenerateDuration time.Duration
	TokensPerSecond  float64
}

// Generate executes autoregressive generation for a text prompt.
func (e *Engine) Generate(prompt string, maxTokens int, params sampler.Params, onToken func(token string) bool) (*GenerateStats, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	tokens := e.Tokenizer.Encode(prompt, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("prompt produced 0 tokens")
	}

	// Security: Prevent context length overflow
	if len(tokens) >= e.Config.SeqLen {
		tokens = tokens[len(tokens)-e.Config.SeqLen+1:]
	}

	// Security: Bound maxTokens to model sequence length
	if maxTokens <= 0 {
		maxTokens = 512
	}
	if maxTokens > e.Config.SeqLen-len(tokens) {
		maxTokens = e.Config.SeqLen - len(tokens)
		if maxTokens <= 0 {
			maxTokens = 1
		}
	}

	kv := e.NewKVCache()
	pos := 0

	// 1. Prefill / Ingest prompt tokens in parallel via batched GEMM
	startPrefill := time.Now()
	if len(tokens) > 1 {
		e.ForwardBatch(tokens, kv)
		pos = len(tokens)
	} else {
		for _, tok := range tokens {
			e.Forward(tok, pos, kv)
			pos++
		}
	}
	prefillDur := time.Since(startPrefill)

	history := append([]int{}, tokens...)
	curTok := tokens[len(tokens)-1]

	// 2. Generation loop
	startGen := time.Now()
	genTokens := 0

	for i := 0; i < maxTokens; i++ {
		logits := e.Forward(curTok, pos, kv)
		next := sampler.SampleToken(logits, history, params)
		pos++
		genTokens++

		if next == e.Config.EosID || next == e.Config.EotID {
			break
		}

		piece := e.Tokenizer.Decode([]int{next})
		if onToken != nil {
			continueGen := onToken(piece)
			if !continueGen {
				break
			}
		}

		history = append(history, next)
		curTok = next
	}
	genDur := time.Since(startGen)

	tps := 0.0
	if genDur.Seconds() > 0 {
		tps = float64(genTokens) / genDur.Seconds()
	}

	return &GenerateStats{
		PromptTokens:     len(tokens),
		GeneratedTokens:  genTokens,
		PrefillDuration:  prefillDur,
		GenerateDuration: genDur,
		TokensPerSecond:  tps,
	}, nil
}

// FormatChat formats messages according to standard LLaMA 3 or ChatML prompt templates.
func (e *Engine) FormatChat(messages []ChatMessage) string {
	var sb strings.Builder
	// Check if vocabulary has LLaMA 3 special tokens
	isLlama3 := e.Tokenizer.EotTokenID != e.Tokenizer.EosTokenID

	if isLlama3 {
		for _, m := range messages {
			sb.WriteString(fmt.Sprintf("<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", m.Role, strings.TrimSpace(m.Content)))
		}
		sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	} else {
		// ChatML fallback
		for _, m := range messages {
			sb.WriteString(fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", m.Role, strings.TrimSpace(m.Content)))
		}
		sb.WriteString("<|im_start|>assistant\n")
	}

	return sb.String()
}
