package engine

import (
	"fmt"
	"go-inference/pkg/gguf"
	"go-inference/pkg/math"
	"go-inference/pkg/metal"
	"go-inference/pkg/reasoning"
	"go-inference/pkg/sampler"
	"go-inference/pkg/tokenizer"
	"math/rand"
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
	MetalLayers              []metal.LayerWeights
	PreallocatedLayers       *metal.PreallocatedLayers
	MetalQwen35Layers        []metal.Qwen35LayerWeights
	PreallocatedQwen35Layers *metal.PreallocatedQwen35Layers
	OutNormBuf               unsafe.Pointer
	OutWeightBuf             unsafe.Pointer
	OutWeightTyp             int
	TokenEmbdBuf             unsafe.Pointer
	TokenEmbdTyp             int
	PrefixCache              *PrefixCache
	mu                       sync.Mutex
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

	// Detect unsupported State Space Model (SSM / Mamba hybrid) architectures
	if arch == "mamba" || arch == "rwkv" {
		reader.Close()
		return nil, fmt.Errorf("architecture '%s' is an unsupported recurrent architecture. go-infer supports dense Transformers and Qwen 3.5 hybrid SSM", arch)
	}

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
	if nextn := int(getParamUint("nextn_predict_layers", 0)); nextn > 0 && numLayers > nextn {
		numLayers -= nextn
	}
	numHeads := int(getParamUint("attention.head_count", 32))
	numKVHeads := int(getParamUint("attention.head_count_kv", uint64(numHeads)))
	seqLen := int(getParamUint("context_length", 2048))
	ropeTheta := float32(getParamFloat("rope.freq_base", 500000.0))
	ropeDim := int(getParamUint("rope.dimension_count", 0))
	eps := float32(getParamFloat("attention.layer_norm_rms_epsilon", 1e-5))
	if eps == 0 {
		eps = float32(getParamFloat("attention.layer_norm_epsilon", 1e-5))
	}

	fullAttnInterval := int(getParamUint("full_attention_interval", 4))
	ssmInnerSize := int(getParamUint("ssm.inner_size", 6144))
	ssmConvKernel := int(getParamUint("ssm.conv_kernel", 4))
	ssmStateSize := int(getParamUint("ssm.state_size", 128))
	ssmGroupCount := int(getParamUint("ssm.group_count", 16))
	ssmTimeStepRank := int(getParamUint("ssm.time_step_rank", 48))

	vocabSize := len(vocab)
	if vocabSize == 0 {
		// Fallback to output or embedding tensor dimensions
		if emb, ok := reader.Header.Tensors["token_embd.weight"]; ok && len(emb.Dimensions) > 1 {
			vocabSize = int(emb.Dimensions[1])
		}
	}

	tok := tokenizer.NewTokenizer(vocab, merges, bosID, eosID)

	addBOS := gguf.GetMetadataBool(meta, "tokenizer.ggml.add_bos_token", false)

	// Cap default active sequence length to 8192 tokens to prevent multi-gigabyte VRAM/RAM exhaustion on 128k+ models
	activeSeqLen := seqLen
	if activeSeqLen <= 0 || activeSeqLen > 8192 {
		activeSeqLen = 8192
	}

	cfg := ModelConfig{
		Architecture:          arch,
		Dim:                   dim,
		HiddenDim:             hiddenDim,
		NumLayers:             numLayers,
		NumHeads:              numHeads,
		NumKVHeads:            numKVHeads,
		VocabSize:             vocabSize,
		SeqLen:                activeSeqLen,
		RopeTheta:             ropeTheta,
		RopeDim:               ropeDim,
		Eps:                   eps,
		BosID:                 bosID,
		EosID:                 eosID,
		EotID:                 tok.EotTokenID,
		AddBOS:                addBOS,
		FullAttentionInterval: fullAttnInterval,
		SSMInnerSize:          ssmInnerSize,
		SSMConvKernel:         ssmConvKernel,
		SSMStateSize:          ssmStateSize,
		SSMGroupCount:         ssmGroupCount,
		SSMTimeStepRank:       ssmTimeStepRank,
	}

	// Try initializing Apple Metal GPU on macOS
	_ = metal.Init()
	if metal.IsAvailable() {
		if cfg.Architecture == "qwen35" {
			ssmChannels := cfg.SSMInnerSize + 2*cfg.SSMGroupCount*cfg.SSMStateSize
			_ = metal.AllocQwen35Buffers(
				cfg.Dim, cfg.HiddenDim, cfg.SSMInnerSize, ssmChannels,
				cfg.SSMTimeStepRank, cfg.SSMStateSize, cfg.KVDim(), cfg.VocabSize,
				cfg.NumLayers, cfg.SeqLen,
			)
		} else {
			allocDim := cfg.Dim
			if cfg.AttnDim() > allocDim {
				allocDim = cfg.AttnDim()
			}
			_ = metal.AllocBuffers(allocDim, cfg.HiddenDim, cfg.KVDim(), cfg.VocabSize, cfg.NumLayers, cfg.SeqLen)
		}
	}

	weights, err := NewWeights(reader)
	if err != nil {
		reader.Close()
		return nil, fmt.Errorf("load weights: %w", err)
	}

	if arch != "qwen35" {
		if _, ok := weights.ResolveTensorName("blk.0.attn_q.weight"); !ok {
			weights.Close()
			reader.Close()
			return nil, fmt.Errorf("model is missing standard attention projection weights ('blk.0.attn_q.weight'); architecture '%s' is not supported", arch)
		}
	} else {
		if _, ok := weights.ResolveTensorName("blk.0.attn_qkv.weight"); !ok {
			weights.Close()
			reader.Close()
			return nil, fmt.Errorf("qwen35 model is missing SSM projection weights ('blk.0.attn_qkv.weight')")
		}
	}

	arena := NewMemoryArena(cfg)
	gemv := math.NewGEMVEngine(numThreads)

	var metalLayers []metal.LayerWeights
	var qwen35Layers []metal.Qwen35LayerWeights
	var preallocatedQwen35Layers *metal.PreallocatedQwen35Layers
	var outNormBuf, outWeightBuf unsafe.Pointer
	var outWeightTyp int
	var tokenEmbdBuf unsafe.Pointer
	var tokenEmbdTyp int

	if metal.IsAvailable() {
		outNormName, _ := weights.ResolveTensorName("output_norm.weight")
		outWeightName, _ := weights.ResolveTensorName("output.weight")
		tokenEmbdName, _ := weights.ResolveTensorName("token_embd.weight")
		outNormBuf = weights.GPUBufs[outNormName]
		outWeightBuf = weights.GPUBufs[outWeightName]
		outWeightTyp = int(weights.Meta[outWeightName].Type)
		tokenEmbdBuf = weights.GPUBufs[tokenEmbdName]
		if meta, ok := weights.Meta[tokenEmbdName]; ok {
			tokenEmbdTyp = int(meta.Type)
		}

		if arch != "qwen35" {
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

				bqName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_q.bias", l))
				bkName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_k.bias", l))
				bvName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_v.bias", l))

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
					BQBuf:       weights.GPUBufs[bqName],
					BKBuf:       weights.GPUBufs[bkName],
					BVBuf:       weights.GPUBufs[bvName],
				}
			}
		} else {
			qwen35Layers = make([]metal.Qwen35LayerWeights, numLayers)
			for l := 0; l < numLayers; l++ {
				attnNormName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_norm.weight", l))
				ffnNormName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.post_attention_norm.weight", l))
				gateName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_gate.weight", l))
				upName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_up.weight", l))
				downName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_down.weight", l))

				isSSM := cfg.IsSSMLayer(l)
				qwen35Layers[l].IsSSM = isSSM
				qwen35Layers[l].AttnNormBuf = weights.GPUBufs[attnNormName]
				qwen35Layers[l].FFNNormBuf = weights.GPUBufs[ffnNormName]
				qwen35Layers[l].FFNGateBuf = weights.GPUBufs[gateName]
				qwen35Layers[l].FFNGateType = int(weights.Meta[gateName].Type)
				qwen35Layers[l].FFNUpBuf = weights.GPUBufs[upName]
				qwen35Layers[l].FFNUpType = int(weights.Meta[upName].Type)
				qwen35Layers[l].FFNDownBuf = weights.GPUBufs[downName]
				qwen35Layers[l].FFNDownType = int(weights.Meta[downName].Type)

				if isSSM {
					ssmGateName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_gate.weight", l))
					ssmQKVName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_qkv.weight", l))
					ssmConvName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ssm_conv1d.weight", l))
					ssmAlphaName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ssm_alpha.weight", l))
					ssmBetaName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ssm_beta.weight", l))
					ssmDTName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ssm_dt.bias", l))
					ssmAName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ssm_a", l))
					ssmNormName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ssm_norm.weight", l))
					ssmOutName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.ssm_out.weight", l))

					qwen35Layers[l].SSMGateBuf = weights.GPUBufs[ssmGateName]
					qwen35Layers[l].SSMGateType = int(weights.Meta[ssmGateName].Type)
					qwen35Layers[l].SSMQKVBuf = weights.GPUBufs[ssmQKVName]
					qwen35Layers[l].SSMQKVType = int(weights.Meta[ssmQKVName].Type)
					qwen35Layers[l].SSMConv1DBuf = weights.GPUBufs[ssmConvName]
					qwen35Layers[l].SSMAlphaBuf = weights.GPUBufs[ssmAlphaName]
					qwen35Layers[l].SSMAlphaType = int(weights.Meta[ssmAlphaName].Type)
					qwen35Layers[l].SSMBetaBuf = weights.GPUBufs[ssmBetaName]
					qwen35Layers[l].SSMBetaType = int(weights.Meta[ssmBetaName].Type)
					qwen35Layers[l].SSMDTBiasBuf = weights.GPUBufs[ssmDTName]
					qwen35Layers[l].SSMABuf = weights.GPUBufs[ssmAName]
					qwen35Layers[l].SSMNormBuf = weights.GPUBufs[ssmNormName]
					qwen35Layers[l].SSMOutBuf = weights.GPUBufs[ssmOutName]
					qwen35Layers[l].SSMOutType = int(weights.Meta[ssmOutName].Type)
				} else {
					wqName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_q.weight", l))
					wkName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_k.weight", l))
					wvName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_v.weight", l))
					woName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_output.weight", l))
					qNormName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_q_norm.weight", l))
					kNormName, _ := weights.ResolveTensorName(fmt.Sprintf("blk.%d.attn_k_norm.weight", l))

					qwen35Layers[l].WQBuf = weights.GPUBufs[wqName]
					qwen35Layers[l].WQType = int(weights.Meta[wqName].Type)
					qwen35Layers[l].WKBuf = weights.GPUBufs[wkName]
					qwen35Layers[l].WKType = int(weights.Meta[wkName].Type)
					qwen35Layers[l].WVBuf = weights.GPUBufs[wvName]
					qwen35Layers[l].WVType = int(weights.Meta[wvName].Type)
					qwen35Layers[l].WOBuf = weights.GPUBufs[woName]
					qwen35Layers[l].WOType = int(weights.Meta[woName].Type)
					qwen35Layers[l].QNormBuf = weights.GPUBufs[qNormName]
					qwen35Layers[l].KNormBuf = weights.GPUBufs[kNormName]
				}
			}
			preallocatedQwen35Layers = metal.NewPreallocatedQwen35Layers(qwen35Layers)
		}
	}

	preallocatedLayers := metal.NewPreallocatedLayers(metalLayers)

	return &Engine{
		Config:                   cfg,
		Reader:                   reader,
		Tokenizer:                tok,
		GEMV:                     gemv,
		Arena:                    arena,
		Weights:                  weights,
		MetalLayers:              metalLayers,
		PreallocatedLayers:       preallocatedLayers,
		MetalQwen35Layers:        qwen35Layers,
		PreallocatedQwen35Layers: preallocatedQwen35Layers,
		OutNormBuf:               outNormBuf,
		OutWeightBuf:             outWeightBuf,
		OutWeightTyp:             outWeightTyp,
		TokenEmbdBuf:             tokenEmbdBuf,
		TokenEmbdTyp:             tokenEmbdTyp,
		PrefixCache:              NewPrefixCache(32),
	}, nil
}

// Close frees model resources and memory mappings.
func (e *Engine) Close() error {
	if e.PreallocatedLayers != nil {
		e.PreallocatedLayers.Free()
		e.PreallocatedLayers = nil
	}
	if e.PreallocatedQwen35Layers != nil {
		e.PreallocatedQwen35Layers.Free()
		e.PreallocatedQwen35Layers = nil
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
	kv := NewKVCache(e.Config.NumLayers, e.Config.SeqLen, e.Config.KVDim())
	if e.Config.Architecture == "qwen35" {
		ssmChannels := e.Config.SSMInnerSize + 2*e.Config.SSMGroupCount*e.Config.SSMStateSize
		kv.InitSSM(e.Config.NumLayers, e.Config.SSMConvKernel, ssmChannels, e.Config.SSMInnerSize, e.Config.SSMStateSize)
	}
	return kv
}

// NewQuantizedKVCache allocates a KV cache with the specified precision type (f32, q8_0, q4_0).
func (e *Engine) NewQuantizedKVCache(kvType KVCacheType) *KVCache {
	kv := NewQuantizedKVCache(e.Config.NumLayers, e.Config.SeqLen, e.Config.KVDim(), kvType)
	if e.Config.Architecture == "qwen35" {
		ssmChannels := e.Config.SSMInnerSize + 2*e.Config.SSMGroupCount*e.Config.SSMStateSize
		kv.InitSSM(e.Config.NumLayers, e.Config.SSMConvKernel, ssmChannels, e.Config.SSMInnerSize, e.Config.SSMStateSize)
	}
	return kv
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
	return e.GenerateWithTools(prompt, maxTokens, params, false, onToken)
}

// GenerateWithTools executes autoregressive generation with optional inline calculator evaluation.
func (e *Engine) GenerateWithTools(prompt string, maxTokens int, params sampler.Params, enableCalc bool, onToken func(token string) bool) (*GenerateStats, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	tokens := e.Tokenizer.Encode(prompt, e.Config.AddBOS)
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

	// Supply vocabulary for structured JSON & grammar constraints if needed
	if params.JSONValidator != nil && len(params.Vocab) == 0 {
		params.Vocab = e.Tokenizer.Vocab
	}
	if params.ReasoningValidator != nil && len(params.Vocab) == 0 {
		params.Vocab = e.Tokenizer.Vocab
	}

	// 1. Check Prefix / Prompt KV-Cache for instant reuse
	var kv *KVCache
	pos := 0
	startPrefill := time.Now()

	if e.PrefixCache != nil {
		matchedLen, cachedKV := e.PrefixCache.FindLongestPrefix(tokens)
		if matchedLen > 0 && cachedKV != nil {
			kv = cachedKV
			pos = matchedLen
		}
	}

	if kv == nil {
		kv = e.NewKVCache()
	}

	// 2. Prefill remaining uncached prompt tokens
	var logits []float32
	if pos < len(tokens) {
		remTokens := tokens[pos:]
		if len(remTokens) > 1 {
			logits = e.ForwardBatchAt(remTokens, pos, kv)
			pos = len(tokens)
		} else {
			for pos < len(tokens) {
				logits = e.Forward(tokens[pos], pos, kv)
				pos++
			}
		}
	} else if len(tokens) > 0 {
		logits = e.Forward(tokens[len(tokens)-1], len(tokens)-1, kv)
	}
	prefillDur := time.Since(startPrefill)

	// Cache the full prompt KV-cache state for future queries
	if e.PrefixCache != nil {
		e.PrefixCache.Store(tokens, kv)
	}

	history := append([]int{}, tokens...)

	// 2. Generation loop
	startGen := time.Now()
	genTokens := 0
	var recentBuffer strings.Builder

	isGreedy := params.Temperature <= 0.0 && params.RepPenalty <= 1.0 && params.JSONValidator == nil && params.ReasoningValidator == nil && !enableCalc

	var next int
	if isGreedy && !enableCalc && maxTokens > 1 {
		// Double-buffered asynchronous GPU pipeline for greedy decoding
		type forwardReq struct {
			tok int
			pos int
		}
		type forwardRes struct {
			tok int
		}

		reqChan := make(chan forwardReq, 1)
		resChan := make(chan forwardRes, 1)

		go func() {
			for req := range reqChan {
				nxt := e.ForwardSample(req.tok, req.pos, kv)
				resChan <- forwardRes{tok: nxt}
			}
		}()
		defer close(reqChan)

		// First token sampled from prefill logits
		next = sampler.SampleToken(logits, history, params)
		for i := 0; i < maxTokens; i++ {
			genTokens++
			if e.isStopToken(next) {
				break
			}

			// Launch asynchronous forward pass on GPU for next token while CPU decodes current token
			hasPendingGPU := false
			if i+1 < maxTokens {
				reqChan <- forwardReq{tok: next, pos: pos}
				pos++
				hasPendingGPU = true
			}

			piece := e.Tokenizer.Decode([]int{next})
			history = append(history, next)

			continueGen := true
			if onToken != nil {
				continueGen = onToken(piece)
			}

			if !continueGen {
				if hasPendingGPU {
					<-resChan
				}
				break
			}

			if hasPendingGPU {
				res := <-resChan
				next = res.tok
			}
		}
	} else {
		for i := 0; i < maxTokens; i++ {
			if i == 0 || !isGreedy {
				next = sampler.SampleToken(logits, history, params)
			}
			genTokens++

			if e.isStopToken(next) {
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

			// 3. Inline Calculator Execution if enabled
			if enableCalc {
				recentBuffer.WriteString(piece)
				bufStr := recentBuffer.String()

				// Check for ```calc\n...\n```
				if (strings.Contains(bufStr, "```calc\n") || strings.Contains(bufStr, "```math\n")) && strings.HasSuffix(bufStr, "\n```") {
					startTag := "```calc\n"
					if !strings.Contains(bufStr, startTag) {
						startTag = "```math\n"
					}
					sIdx := strings.Index(bufStr, startTag) + len(startTag)
					eIdx := len(bufStr) - len("\n```")
					if eIdx > sIdx {
						expr := strings.TrimSpace(bufStr[sIdx:eIdx])
						if val, err := reasoning.ExecuteMathTool(expr); err == nil {
							inject := fmt.Sprintf("\n--> result: %s\n```\n", val)
							if onToken != nil {
								onToken(inject)
							}
							injectTokens := e.Tokenizer.Encode(inject, false)
							for _, it := range injectTokens {
								logits = e.Forward(it, pos, kv)
								pos++
								history = append(history, it)
							}
							recentBuffer.Reset()
							continue
						}
					}
					recentBuffer.Reset()
				} else if strings.Contains(bufStr, "<<calc:") && strings.HasSuffix(bufStr, ">>") {
					sIdx := strings.Index(bufStr, "<<calc:") + len("<<calc:")
					eIdx := len(bufStr) - len(">>")
					if eIdx > sIdx {
						expr := strings.TrimSpace(bufStr[sIdx:eIdx])
						if !strings.Contains(expr, "=") {
							if val, err := reasoning.ExecuteMathTool(expr); err == nil {
								inject := fmt.Sprintf(" = %s>>", val)
								if onToken != nil {
									onToken(inject)
								}
								injectTokens := e.Tokenizer.Encode(inject, false)
								for _, it := range injectTokens {
									logits = e.Forward(it, pos, kv)
									pos++
									history = append(history, it)
								}
								recentBuffer.Reset()
								continue
							}
						}
					}
					recentBuffer.Reset()
				}
			}

			if isGreedy {
				next = e.ForwardSample(next, pos, kv)
			} else {
				logits = e.Forward(next, pos, kv)
			}
			pos++
		}
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

// GenerateConsensus runs self-consistency majority voting across numSamples parallel branches.
func (e *Engine) GenerateConsensus(prompt string, numSamples int, maxTokens int, params sampler.Params, onCandidate func(cand *reasoning.CandidateAnswer)) (*reasoning.ConsensusResult, *GenerateStats, error) {
	return e.GenerateConsensusWithStream(prompt, numSamples, maxTokens, params, nil, onCandidate)
}

// GenerateConsensusWithStream runs self-consistency majority voting with real-time candidate token streaming.
func (e *Engine) GenerateConsensusWithStream(prompt string, numSamples int, maxTokens int, params sampler.Params, onToken func(sampleIdx int, piece string), onCandidate func(cand *reasoning.CandidateAnswer)) (*reasoning.ConsensusResult, *GenerateStats, error) {
	if numSamples <= 1 {
		numSamples = 3
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	tokens := e.Tokenizer.Encode(prompt, e.Config.AddBOS)
	if len(tokens) == 0 {
		return nil, nil, fmt.Errorf("prompt produced 0 tokens")
	}

	if len(tokens) >= e.Config.SeqLen {
		tokens = tokens[len(tokens)-e.Config.SeqLen+1:]
	}

	if maxTokens <= 0 {
		maxTokens = 512
	}
	if maxTokens > e.Config.SeqLen-len(tokens) {
		maxTokens = e.Config.SeqLen - len(tokens)
		if maxTokens <= 0 {
			maxTokens = 1
		}
	}

	// 1. Prefill prompt once
	startPrefill := time.Now()
	var baseKV *KVCache
	pos := 0

	if e.PrefixCache != nil {
		matchedLen, cachedKV := e.PrefixCache.FindLongestPrefix(tokens)
		if matchedLen > 0 && cachedKV != nil {
			baseKV = cachedKV
			pos = matchedLen
		}
	}

	if baseKV == nil {
		baseKV = e.NewKVCache()
	}

	var baseLogits []float32
	if pos < len(tokens) {
		remTokens := tokens[pos:]
		if len(remTokens) > 1 {
			baseLogits = e.ForwardBatchAt(remTokens, pos, baseKV)
			pos = len(tokens)
		} else {
			for pos < len(tokens) {
				baseLogits = e.Forward(tokens[pos], pos, baseKV)
				pos++
			}
		}
	} else if len(tokens) > 0 {
		baseLogits = e.Forward(tokens[len(tokens)-1], len(tokens)-1, baseKV)
	}
	prefillDur := time.Since(startPrefill)

	// Save a detached copy of baseLogits so Forward() in sampling loop doesn't mutate it
	savedBaseLogits := make([]float32, len(baseLogits))
	copy(savedBaseLogits, baseLogits)

	if e.PrefixCache != nil {
		e.PrefixCache.Store(tokens, baseKV)
	}

	// 2. Sample independent branches using cloned KV-cache states
	startGen := time.Now()
	totalGenTokens := 0
	candidates := make([]reasoning.CandidateAnswer, numSamples)

	for s := 0; s < numSamples; s++ {
		sampleKV := baseKV.Clone()
		sampleParams := params
		sampleParams.Rand = rand.New(rand.NewSource(time.Now().UnixNano() + int64(s*10007)))

		history := append([]int{}, tokens...)
		samplePos := len(tokens)
		logits := make([]float32, len(savedBaseLogits))
		copy(logits, savedBaseLogits)

		var outputBuilder strings.Builder

		for i := 0; i < maxTokens; i++ {
			next := sampler.SampleToken(logits, history, sampleParams)
			samplePos++
			totalGenTokens++

			if e.isStopToken(next) {
				break
			}

			piece := e.Tokenizer.Decode([]int{next})
			outputBuilder.WriteString(piece)
			if onToken != nil {
				onToken(s, piece)
			}

			history = append(history, next)
			logits = e.Forward(next, samplePos-1, sampleKV)
		}

		fullText := outputBuilder.String()
		thinking, answer := reasoning.ExtractThinking(fullText)
		rawAnswer := reasoning.ExtractAnswer(answer)
		if rawAnswer == "" {
			rawAnswer = reasoning.ExtractAnswer(fullText)
		}

		cand := reasoning.CandidateAnswer{
			Index:      s,
			RawAnswer:  rawAnswer,
			NormAnswer: reasoning.NormalizeAnswer(rawAnswer),
			Thinking:   thinking,
			FullOutput: fullText,
		}
		candidates[s] = cand
		if onCandidate != nil {
			onCandidate(&cand)
		}
	}

	genDur := time.Since(startGen)
	tps := 0.0
	if genDur.Seconds() > 0 {
		tps = float64(totalGenTokens) / genDur.Seconds()
	}

	consensus := reasoning.EvaluateConsensus(candidates)
	stats := &GenerateStats{
		PromptTokens:     len(tokens),
		GeneratedTokens:  totalGenTokens,
		PrefillDuration:  prefillDur,
		GenerateDuration: genDur,
		TokensPerSecond:  tps,
	}

	return consensus, stats, nil
}

func (e *Engine) isStopToken(tok int) bool {
	if tok == e.Config.EosID || tok == e.Config.EotID {
		return true
	}
	if e.Tokenizer != nil {
		if id, ok := e.Tokenizer.TokenToID["<|im_end|>"]; ok && tok == id {
			return true
		}
		if id, ok := e.Tokenizer.TokenToID["<|endoftext|>"]; ok && tok == id {
			return true
		}
		if id, ok := e.Tokenizer.TokenToID["<|eot_id|>"]; ok && tok == id {
			return true
		}
		if id, ok := e.Tokenizer.TokenToID["<｜end▁of▁sentence｜>"]; ok && tok == id {
			return true
		}
		if id, ok := e.Tokenizer.TokenToID["<｜User｜>"]; ok && tok == id {
			return true
		}
		if id, ok := e.Tokenizer.TokenToID["<｜Assistant｜>"]; ok && tok == id {
			return true
		}
		if id, ok := e.Tokenizer.TokenToID["<｜begin▁of▁sentence｜>"]; ok && tok == id {
			return true
		}
	}
	return false
}

// FormatChat formats messages according to standard LLaMA 3, DeepSeek, or ChatML prompt templates.
func (e *Engine) FormatChat(messages []ChatMessage) string {
	var sb strings.Builder

	// 1. Check DeepSeek R1 / V3
	if _, hasDeepSeek := e.Tokenizer.TokenToID["<｜User｜>"]; hasDeepSeek {
		for _, m := range messages {
			if m.Role == "user" {
				sb.WriteString(fmt.Sprintf("<｜User｜>%s\n", strings.TrimSpace(m.Content)))
			} else if m.Role == "assistant" {
				sb.WriteString(fmt.Sprintf("<｜Assistant｜>%s<｜end▁of▁sentence｜>\n", strings.TrimSpace(m.Content)))
			}
		}
		sb.WriteString("<｜Assistant｜>\n<think>\n")
		return sb.String()
	}

	// 2. Check LLaMA 3
	_, hasLlamaHeader := e.Tokenizer.TokenToID["<|start_header_id|>"]
	_, hasChatML := e.Tokenizer.TokenToID["<|im_start|>"]
	isLlama3 := hasLlamaHeader || (!hasChatML && e.Tokenizer.EotTokenID != e.Tokenizer.EosTokenID)

	if isLlama3 {
		for _, m := range messages {
			sb.WriteString(fmt.Sprintf("<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", m.Role, strings.TrimSpace(m.Content)))
		}
		sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	} else {
		// 3. ChatML fallback (Qwen, Yi, etc.)
		for _, m := range messages {
			sb.WriteString(fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", m.Role, strings.TrimSpace(m.Content)))
		}
		sb.WriteString("<|im_start|>assistant\n")
	}

	return sb.String()
}
