package engine

import (
	"fmt"
	"go-inference/pkg/gguf"
	"go-inference/pkg/math"
	"go-inference/pkg/metal"
	"go-inference/pkg/quant"
	stdmath "math"
	"sync"
)

// Forward runs a complete autoregressive transformer forward pass for a single token at position pos.
func (e *Engine) Forward(token int, pos int, kv *KVCache) []float32 {
	cfg := e.Config
	headDim := cfg.HeadDim()
	kvDim := cfg.KVDim()
	kvMul := cfg.KVMul()
	a := e.Arena
	w := e.Weights
	isMetal := metal.IsAvailable()

	// 1. Embedding lookup (GPU-accelerated when Q4_K token_embd buffer is resident)
	useGPUEmbed := isMetal && cfg.Architecture == "qwen35" && e.TokenEmbdBuf != nil && e.TokenEmbdTyp == int(gguf.GGMLTypeQ4_K) && e.PreallocatedQwen35Layers != nil && e.OutNormBuf != nil && e.OutWeightBuf != nil
	if !useGPUEmbed {
		w.ExtractEmbedding(token, a.X, cfg.Dim)
	}

	activeContext := pos + 1
	if activeContext > kv.MaxSeq {
		activeContext = kv.MaxSeq
	}
	attnScale := float32(1.0 / math_sqrt(float64(headDim)))

	// Specialized Hybrid SSM path for Qwen 3.5
	if cfg.Architecture == "qwen35" {
		// Fast path: Single CGo call for all hybrid layers with 100% GPU-resident activations
		if isMetal && e.PreallocatedQwen35Layers != nil && e.OutNormBuf != nil && e.OutWeightBuf != nil {
			ssmChannels := cfg.SSMInnerSize + 2*cfg.SSMGroupCount*cfg.SSMStateSize
			tp := metal.Qwen35TransformerParams{
				InitialX:           a.X,
				TokenEmbdBuf:       e.TokenEmbdBuf,
				TokenID:            token,
				OutLogits:          a.Logits,
				PreallocatedLayers: e.PreallocatedQwen35Layers,
				OutputNormBuf:      e.OutNormBuf,
				OutputWeightBuf:    e.OutWeightBuf,
				OutputWeightType:   e.OutWeightTyp,
				NumLayers:          cfg.NumLayers,
				Dim:                cfg.Dim,
				HiddenDim:          cfg.HiddenDim,
				SSMInner:           cfg.SSMInnerSize,
				SSMChannels:        ssmChannels,
				SSMStateSize:       cfg.SSMStateSize,
				SSMGroups:          cfg.SSMGroupCount,
				SSMRank:            cfg.SSMTimeStepRank,
				KVDim:              kvDim,
				VocabSize:          cfg.VocabSize,
				NumHeads:           cfg.NumHeads,
				NumKVHeads:         cfg.NumKVHeads,
				HeadDim:            headDim,
				RopeDim:            cfg.RopeDim,
				Pos:                pos,
				Slot:               pos % kv.MaxSeq,
				MaxSeq:             kv.MaxSeq,
				ActiveContext:      activeContext,
				NormEps:            cfg.Eps,
				RopeTheta:          cfg.RopeTheta,
				AttnScale:          attnScale,
			}
			if err := metal.ForwardQwen35(&tp); err == nil {
				return a.Logits
			}
			if useGPUEmbed {
				w.ExtractEmbedding(token, a.X, cfg.Dim)
			}
		}

		if isMetal {
			metal.BeginBatch()
			defer metal.EndBatch()
		}
		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.IsSSMLayer(l) {
				e.forwardQwen35SSMLayer(l, a, kv, cfg)
			} else {
				e.forwardQwen35AttentionLayer(l, pos, activeContext, a, kv, cfg)
			}
		}

		// Final RMSNorm
		outputNormW := w.Get1DWeight("output_norm.weight", cfg.Dim)
		if !isMetal || metal.RMSNorm(a.XB, a.X, outputNormW, cfg.Dim, cfg.Eps) != nil {
			math.RMSNorm(a.XB, a.X, outputNormW, cfg.Eps)
		}

		// Final Logits projection
		w.MatMul(e.GEMV, a.Logits, a.XB, "output.weight", cfg.VocabSize, cfg.Dim)
		return a.Logits
	}

	// Fast path: Single CGo call for all 40 layers with 100% GPU-resident activations
	if isMetal && len(e.MetalLayers) == cfg.NumLayers && e.OutNormBuf != nil && e.OutWeightBuf != nil {
		tp := metal.TransformerParams{
			InitialX:           a.X,
			OutLogits:          a.Logits,
			Layers:             e.MetalLayers,
			PreallocatedLayers: e.PreallocatedLayers,
			OutputNormBuf:      e.OutNormBuf,
			OutputWeightBuf:    e.OutWeightBuf,
			OutputWeightType:   e.OutWeightTyp,
			NumLayers:        cfg.NumLayers,
			Dim:              cfg.Dim,
			HiddenDim:        cfg.HiddenDim,
			KVDim:            kvDim,
			VocabSize:        cfg.VocabSize,
			NumHeads:         cfg.NumHeads,
			NumKVHeads:       cfg.NumKVHeads,
			HeadDim:          headDim,
			Pos:              pos,
			Slot:             pos % kv.MaxSeq,
			MaxSeq:           kv.MaxSeq,
			ActiveContext:    activeContext,
			NormEps:          cfg.Eps,
			RopeTheta:        cfg.RopeTheta,
			AttnScale:        attnScale,
		}
		if err := metal.ForwardTransformer(&tp); err == nil {
			return a.Logits
		} else {
			fmt.Printf("ForwardTransformer err: %v\n", err)
		}
	}

	if isMetal {
		metal.BeginBatch()
		defer metal.EndBatch()
	}

	// 2. Transformer layers
	for l := 0; l < cfg.NumLayers; l++ {
		if isMetal {
			attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), cfg.Dim)
			ffnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.ffn_norm.weight", l), cfg.Dim)

			wqName, _ := w.ResolveTensorName(fmt.Sprintf("blk.%d.attn_q.weight", l))
			wkName, _ := w.ResolveTensorName(fmt.Sprintf("blk.%d.attn_k.weight", l))
			wvName, _ := w.ResolveTensorName(fmt.Sprintf("blk.%d.attn_v.weight", l))
			woName, _ := w.ResolveTensorName(fmt.Sprintf("blk.%d.attn_output.weight", l))
			gateName, _ := w.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_gate.weight", l))
			upName, _ := w.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_up.weight", l))
			downName, _ := w.ResolveTensorName(fmt.Sprintf("blk.%d.ffn_down.weight", l))

			lp := metal.LayerParams{
				X:             a.X,
				XNorm:         a.XB,
				Q:             a.Q,
				K:             a.K,
				V:             a.V,
				AttnOut:       a.AttnOut,
				AttnProj:      a.AttnProj,
				FFNGate:       a.Gate,
				FFNUp:         a.Up,
				FFNDown:       a.FFNDown,
				AttnNorm:      attnNormW,
				FFNNorm:       ffnNormW,
				WQBuf:         w.GPUBufs[wqName],
				WQType:        int(w.Meta[wqName].Type),
				WKBuf:         w.GPUBufs[wkName],
				WKType:        int(w.Meta[wkName].Type),
				WVBuf:         w.GPUBufs[wvName],
				WVType:        int(w.Meta[wvName].Type),
				WOBuf:         w.GPUBufs[woName],
				WOType:        int(w.Meta[woName].Type),
				FFNGateBuf:    w.GPUBufs[gateName],
				FFNGateType:   int(w.Meta[gateName].Type),
				FFNUpBuf:      w.GPUBufs[upName],
				FFNUpType:     int(w.Meta[upName].Type),
				FFNDownBuf:    w.GPUBufs[downName],
				FFNDownType:   int(w.Meta[downName].Type),
				LayerIdx:      l,
				Dim:           cfg.Dim,
				HiddenDim:     cfg.HiddenDim,
				KVDim:         kvDim,
				NumHeads:      cfg.NumHeads,
				NumKVHeads:    cfg.NumKVHeads,
				HeadDim:       headDim,
				Pos:           pos,
				Slot:          pos % kv.MaxSeq,
				MaxSeq:        kv.MaxSeq,
				ActiveContext: activeContext,
				NormEps:       cfg.Eps,
				RopeTheta:     cfg.RopeTheta,
				AttnScale:     attnScale,
			}

			if err := metal.ForwardLayer(&lp); err == nil {
				continue
			}
		}

		// Fallback CPU path
		attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), cfg.Dim)
		math.RMSNorm(a.XB, a.X, attnNormW, cfg.Eps)

		// Q, K, V Projections
		w.MatMul(e.GEMV, a.Q, a.XB, fmt.Sprintf("blk.%d.attn_q.weight", l), cfg.Dim, cfg.Dim)
		w.MatMul(e.GEMV, a.K, a.XB, fmt.Sprintf("blk.%d.attn_k.weight", l), kvDim, cfg.Dim)
		w.MatMul(e.GEMV, a.V, a.XB, fmt.Sprintf("blk.%d.attn_v.weight", l), kvDim, cfg.Dim)

		// Add Q, K, V biases if present (e.g. Qwen 2 / 2.5)
		if bq := w.Get1DBias(fmt.Sprintf("blk.%d.attn_q.bias", l)); bq != nil {
			for i, b := range bq {
				a.Q[i] += b
			}
		}
		if bk := w.Get1DBias(fmt.Sprintf("blk.%d.attn_k.bias", l)); bk != nil {
			for i, b := range bk {
				a.K[i] += b
			}
		}
		if bv := w.Get1DBias(fmt.Sprintf("blk.%d.attn_v.bias", l)); bv != nil {
			for i, b := range bv {
				a.V[i] += b
			}
		}

		// Apply RoPE
		for h := 0; h < cfg.NumHeads; h++ {
			math.ApplyRoPE(a.Q[h*headDim:(h+1)*headDim], pos, headDim, cfg.RopeTheta)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			math.ApplyRoPE(a.K[h*headDim:(h+1)*headDim], pos, headDim, cfg.RopeTheta)
		}

		// Store into KV cache
		slot := pos % kv.MaxSeq
		cacheOffset := slot * kvDim
		copy(kv.Key[l][cacheOffset:cacheOffset+kvDim], a.K)
		copy(kv.Value[l][cacheOffset:cacheOffset+kvDim], a.V)

		// Multi-Head Attention with GQA support
		for i := range a.AttnOut {
			a.AttnOut[i] = 0
		}

		for h := 0; h < cfg.NumHeads; h++ {
			qHead := a.Q[h*headDim : (h+1)*headDim]
			kvHeadIdx := h / kvMul
			scores := a.AttnScores[:activeContext]

			for t := 0; t < activeContext; t++ {
				kHead := kv.Key[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
				scores[t] = quant.DotVecF32(qHead, kHead) * attnScale
			}

			math.Softmax(scores)

			outHead := a.AttnOut[h*headDim : (h+1)*headDim]
			for t := 0; t < activeContext; t++ {
				vHead := kv.Value[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
				weight := scores[t]
				for d := 0; d < headDim; d++ {
					outHead[d] += weight * vHead[d]
				}
			}
		}

		// Attention Output Projection & Residual
		w.MatMul(e.GEMV, a.AttnProj, a.AttnOut, fmt.Sprintf("blk.%d.attn_output.weight", l), cfg.Dim, cfg.Dim)
		for i := 0; i < cfg.Dim; i++ {
			a.X[i] += a.AttnProj[i]
		}

		// Feed-Forward (SwiGLU MLP)
		ffnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.ffn_norm.weight", l), cfg.Dim)
		math.RMSNorm(a.XB, a.X, ffnNormW, cfg.Eps)

		w.MatMul(e.GEMV, a.Gate, a.XB, fmt.Sprintf("blk.%d.ffn_gate.weight", l), cfg.HiddenDim, cfg.Dim)
		w.MatMul(e.GEMV, a.Up, a.XB, fmt.Sprintf("blk.%d.ffn_up.weight", l), cfg.HiddenDim, cfg.Dim)
		math.SwiGLU(a.Gate, a.Up, cfg.HiddenDim)

		w.MatMul(e.GEMV, a.FFNDown, a.Gate, fmt.Sprintf("blk.%d.ffn_down.weight", l), cfg.Dim, cfg.HiddenDim)
		for i := 0; i < cfg.Dim; i++ {
			a.X[i] += a.FFNDown[i]
		}
	}

	// 3. Final RMSNorm
	outputNormW := w.Get1DWeight("output_norm.weight", cfg.Dim)
	if !isMetal || metal.RMSNorm(a.XB, a.X, outputNormW, cfg.Dim, cfg.Eps) != nil {
		math.RMSNorm(a.XB, a.X, outputNormW, cfg.Eps)
	}

	// 4. Final Logits projection
	w.MatMul(e.GEMV, a.Logits, a.XB, "output.weight", cfg.VocabSize, cfg.Dim)
	return a.Logits
}

// ForwardSample executes a full forward pass and performs GPU-resident argmax sampling.
// It eliminates CPU logit transfer and scanning overhead (saving ~50ms per token).
func (e *Engine) ForwardSample(token int, pos int, kv *KVCache) int {
	cfg := e.Config
	headDim := cfg.HeadDim()
	kvDim := cfg.KVDim()
	a := e.Arena
	w := e.Weights
	isMetal := metal.IsAvailable()

	// 1. Embedding lookup (GPU-accelerated when Q4_K token_embd buffer is resident)
	useGPUEmbed := isMetal && cfg.Architecture == "qwen35" && e.TokenEmbdBuf != nil && e.TokenEmbdTyp == int(gguf.GGMLTypeQ4_K) && e.PreallocatedQwen35Layers != nil && e.OutNormBuf != nil && e.OutWeightBuf != nil
	if !useGPUEmbed {
		w.ExtractEmbedding(token, a.X, cfg.Dim)
	}

	activeContext := pos + 1
	if activeContext > kv.MaxSeq {
		activeContext = kv.MaxSeq
	}
	attnScale := float32(1.0 / math_sqrt(float64(headDim)))

	if cfg.Architecture == "qwen35" {
		if isMetal && e.PreallocatedQwen35Layers != nil && e.OutNormBuf != nil && e.OutWeightBuf != nil {
			ssmChannels := cfg.SSMInnerSize + 2*cfg.SSMGroupCount*cfg.SSMStateSize
			var nextToken uint32
			tp := metal.Qwen35TransformerParams{
				InitialX:           a.X,
				TokenEmbdBuf:       e.TokenEmbdBuf,
				TokenID:            token,
				OutLogits:          nil,
				OutToken:           &nextToken,
				PreallocatedLayers: e.PreallocatedQwen35Layers,
				OutputNormBuf:      e.OutNormBuf,
				OutputWeightBuf:    e.OutWeightBuf,
				OutputWeightType:   e.OutWeightTyp,
				NumLayers:          cfg.NumLayers,
				Dim:                cfg.Dim,
				HiddenDim:          cfg.HiddenDim,
				SSMInner:           cfg.SSMInnerSize,
				SSMChannels:        ssmChannels,
				SSMStateSize:       cfg.SSMStateSize,
				SSMGroups:          cfg.SSMGroupCount,
				SSMRank:            cfg.SSMTimeStepRank,
				KVDim:              kvDim,
				VocabSize:          cfg.VocabSize,
				NumHeads:           cfg.NumHeads,
				NumKVHeads:         cfg.NumKVHeads,
				HeadDim:            headDim,
				RopeDim:            cfg.RopeDim,
				Pos:                pos,
				Slot:               pos % kv.MaxSeq,
				MaxSeq:             kv.MaxSeq,
				ActiveContext:      activeContext,
				NormEps:            cfg.Eps,
				RopeTheta:          cfg.RopeTheta,
				AttnScale:          attnScale,
			}
			if err := metal.ForwardQwen35(&tp); err == nil {
				return int(nextToken)
			}
			if useGPUEmbed {
				w.ExtractEmbedding(token, a.X, cfg.Dim)
			}
		}
	} else if isMetal && e.PreallocatedLayers != nil && e.OutNormBuf != nil && e.OutWeightBuf != nil {
		var nextToken uint32
		tp := metal.TransformerParams{
			InitialX:           a.X,
			OutLogits:          nil,
			OutToken:           &nextToken,
			PreallocatedLayers: e.PreallocatedLayers,
			OutputNormBuf:      e.OutNormBuf,
			OutputWeightBuf:    e.OutWeightBuf,
			OutputWeightType:   e.OutWeightTyp,
			NumLayers:          cfg.NumLayers,
			Dim:                cfg.Dim,
			HiddenDim:          cfg.HiddenDim,
			KVDim:              kvDim,
			VocabSize:          cfg.VocabSize,
			NumHeads:           cfg.NumHeads,
			NumKVHeads:         cfg.NumKVHeads,
			HeadDim:            headDim,
			Pos:                pos,
			Slot:               pos % kv.MaxSeq,
			MaxSeq:             kv.MaxSeq,
			ActiveContext:      activeContext,
			NormEps:            cfg.Eps,
			RopeTheta:          cfg.RopeTheta,
			AttnScale:          attnScale,
		}
		if err := metal.ForwardTransformer(&tp); err == nil {
			return int(nextToken)
		}
	}

	logits := e.Forward(token, pos, kv)
	best := 0
	maxLogit := logits[0]
	for i, l := range logits {
		if l > maxLogit {
			maxLogit = l
			best = i
		}
	}
	return best
}


// ForwardBatch evaluates a batch of tokens in parallel during prompt prefill starting at pos 0.
func (e *Engine) ForwardBatch(tokens []int, kv *KVCache) []float32 {
	return e.ForwardBatchAt(tokens, 0, kv)
}

// ForwardBatchAt evaluates a batch of tokens in parallel during prompt prefill starting at startPos.
func (e *Engine) ForwardBatchAt(tokens []int, startPos int, kv *KVCache) []float32 {
	batchSize := len(tokens)
	if batchSize == 0 {
		return nil
	}
	if batchSize == 1 {
		return e.Forward(tokens[0], startPos, kv)
	}

	cfg := e.Config

	if cfg.Architecture == "qwen35" {
		if metal.IsAvailable() && e.PreallocatedQwen35Layers != nil && e.OutNormBuf != nil && e.OutWeightBuf != nil {
			var logits []float32
			for i, tok := range tokens {
				logits = e.Forward(tok, startPos+i, kv)
			}
			return logits
		}
		return e.forwardBatchQwen35(tokens, startPos, kv, cfg)
	}

	headDim := cfg.HeadDim()
	kvDim := cfg.KVDim()
	kvMul := cfg.KVMul()
	w := e.Weights

	isMetal := metal.IsAvailable()

	// 1. Embedding matrix (batchSize x Dim)
	batchX := make([]float32, batchSize*cfg.Dim)
	if isMetal && e.TokenEmbdBuf != nil && e.TokenEmbdTyp == int(gguf.GGMLTypeQ4_K) {
		if err := metal.EmbedLookupQ4KBatch(batchX, e.TokenEmbdBuf, tokens, cfg.Dim); err != nil {
			for i, tok := range tokens {
				w.ExtractEmbedding(tok, batchX[i*cfg.Dim:(i+1)*cfg.Dim], cfg.Dim)
			}
		}
	} else {
		for i, tok := range tokens {
			w.ExtractEmbedding(tok, batchX[i*cfg.Dim:(i+1)*cfg.Dim], cfg.Dim)
		}
	}

	batchXB := make([]float32, batchSize*cfg.Dim)
	batchQ := make([]float32, batchSize*cfg.Dim)
	batchK := make([]float32, batchSize*kvDim)
	batchV := make([]float32, batchSize*kvDim)
	batchAttnOut := make([]float32, batchSize*cfg.Dim)
	batchAttnProj := make([]float32, batchSize*cfg.Dim)
	batchGate := make([]float32, batchSize*cfg.HiddenDim)
	var batchUp []float32
	batchFFNDown := make([]float32, batchSize*cfg.Dim)

	attnScale := float32(1.0 / math_sqrt(float64(headDim)))

	for l := 0; l < cfg.NumLayers; l++ {
		attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), cfg.Dim)
		if !isMetal || metal.RMSNormBatch(batchXB, batchX, attnNormW, cfg.Dim, cfg.Eps, batchSize) != nil {
			for b := 0; b < batchSize; b++ {
				math.RMSNorm(batchXB[b*cfg.Dim:(b+1)*cfg.Dim], batchX[b*cfg.Dim:(b+1)*cfg.Dim], attnNormW, cfg.Eps)
			}
		}

		// Batched Q, K, V projections
		if isMetal {
			metal.BeginBatch()
		}
		if !w.MatMulBatch(batchQ, batchXB, fmt.Sprintf("blk.%d.attn_q.weight", l), batchSize, cfg.Dim, cfg.Dim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchQ[b*cfg.Dim:(b+1)*cfg.Dim], batchXB[b*cfg.Dim:(b+1)*cfg.Dim], fmt.Sprintf("blk.%d.attn_q.weight", l), cfg.Dim, cfg.Dim)
			}
		}
		if !w.MatMulBatch(batchK, batchXB, fmt.Sprintf("blk.%d.attn_k.weight", l), batchSize, kvDim, cfg.Dim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchK[b*kvDim:(b+1)*kvDim], batchXB[b*cfg.Dim:(b+1)*cfg.Dim], fmt.Sprintf("blk.%d.attn_k.weight", l), kvDim, cfg.Dim)
			}
		}
		if !w.MatMulBatch(batchV, batchXB, fmt.Sprintf("blk.%d.attn_v.weight", l), batchSize, kvDim, cfg.Dim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchV[b*kvDim:(b+1)*kvDim], batchXB[b*cfg.Dim:(b+1)*cfg.Dim], fmt.Sprintf("blk.%d.attn_v.weight", l), kvDim, cfg.Dim)
			}
		}
		if isMetal {
			metal.EndBatch()
		}

		// Add Q, K, V biases if present (e.g. Qwen 2 / 2.5)
		if bq := w.Get1DBias(fmt.Sprintf("blk.%d.attn_q.bias", l)); bq != nil {
			for b := 0; b < batchSize; b++ {
				qBase := b * cfg.Dim
				for i, v := range bq {
					batchQ[qBase+i] += v
				}
			}
		}
		if bk := w.Get1DBias(fmt.Sprintf("blk.%d.attn_k.bias", l)); bk != nil {
			for b := 0; b < batchSize; b++ {
				kBase := b * kvDim
				for i, v := range bk {
					batchK[kBase+i] += v
				}
			}
		}
		if bv := w.Get1DBias(fmt.Sprintf("blk.%d.attn_v.bias", l)); bv != nil {
			for b := 0; b < batchSize; b++ {
				vBase := b * kvDim
				for i, v := range bv {
					batchV[vBase+i] += v
				}
			}
		}

		// Apply RoPE & write into KV cache for each position
		for b := 0; b < batchSize; b++ {
			pos := startPos + b
			qBase := b * cfg.Dim
			kBase := b * kvDim
			vBase := b * kvDim

			for h := 0; h < cfg.NumHeads; h++ {
				math.ApplyRoPE(batchQ[qBase+h*headDim:qBase+(h+1)*headDim], pos, headDim, cfg.RopeTheta)
			}
			for h := 0; h < cfg.NumKVHeads; h++ {
				math.ApplyRoPE(batchK[kBase+h*headDim:kBase+(h+1)*headDim], pos, headDim, cfg.RopeTheta)
			}

			slot := pos % kv.MaxSeq
			cacheOffset := slot * kvDim
			copy(kv.Key[l][cacheOffset:cacheOffset+kvDim], batchK[kBase:kBase+kvDim])
			copy(kv.Value[l][cacheOffset:cacheOffset+kvDim], batchV[vBase:vBase+kvDim])
		}

		// Multi-head Causal Attention
		for b := 0; b < batchSize; b++ {
			pos := startPos + b
			activeContext := pos + 1
			if activeContext > kv.MaxSeq {
				activeContext = kv.MaxSeq
			}
			qBase := b * cfg.Dim
			outBase := b * cfg.Dim
			scores := make([]float32, activeContext)

			for h := 0; h < cfg.NumHeads; h++ {
				qHead := batchQ[qBase+h*headDim : qBase+(h+1)*headDim]
				kvHeadIdx := h / kvMul

				for t := 0; t < activeContext; t++ {
					kHead := kv.Key[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
					scores[t] = quant.DotVecF32(qHead, kHead) * attnScale
				}
				math.Softmax(scores)

				outHead := batchAttnOut[outBase+h*headDim : outBase+(h+1)*headDim]
				for t := 0; t < activeContext; t++ {
					vHead := kv.Value[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
					weight := scores[t]
					for d := 0; d < headDim; d++ {
						outHead[d] += weight * vHead[d]
					}
				}
			}
		}

		// Attention Output Projection
		if !w.MatMulBatch(batchAttnProj, batchAttnOut, fmt.Sprintf("blk.%d.attn_output.weight", l), batchSize, cfg.Dim, cfg.Dim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchAttnProj[b*cfg.Dim:(b+1)*cfg.Dim], batchAttnOut[b*cfg.Dim:(b+1)*cfg.Dim], fmt.Sprintf("blk.%d.attn_output.weight", l), cfg.Dim, cfg.Dim)
			}
		}

		// Residual Add & FFN Norm (fused when GPU is available)
		ffnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.ffn_norm.weight", l), cfg.Dim)
		if !isMetal || metal.ResidualRMSNormBatch(batchX, batchAttnProj, batchXB, ffnNormW, cfg.Dim, cfg.Eps, batchSize) != nil {
			if isMetal {
				metal.AddResidual(batchX, batchAttnProj, batchSize*cfg.Dim)
			} else {
				for i := range batchX {
					batchX[i] += batchAttnProj[i]
				}
			}
			for b := 0; b < batchSize; b++ {
				math.RMSNorm(batchXB[b*cfg.Dim:(b+1)*cfg.Dim], batchX[b*cfg.Dim:(b+1)*cfg.Dim], ffnNormW, cfg.Eps)
			}
		}

		// Batched FFN (Fused Gate + Up + SwiGLU when available)
		gateName := fmt.Sprintf("blk.%d.ffn_gate.weight", l)
		upName := fmt.Sprintf("blk.%d.ffn_up.weight", l)
		downName := fmt.Sprintf("blk.%d.ffn_down.weight", l)

		if isMetal {
			metal.BeginBatch()
		}
		if !w.MatMulFusedGateUpBatch(batchGate, batchXB, gateName, upName, batchSize, cfg.HiddenDim, cfg.Dim) {
			if batchUp == nil {
				batchUp = make([]float32, batchSize*cfg.HiddenDim)
			}
			if !w.MatMulBatch(batchGate, batchXB, gateName, batchSize, cfg.HiddenDim, cfg.Dim) {
				for b := 0; b < batchSize; b++ {
					w.MatMul(e.GEMV, batchGate[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], batchXB[b*cfg.Dim:(b+1)*cfg.Dim], gateName, cfg.HiddenDim, cfg.Dim)
				}
			}
			if !w.MatMulBatch(batchUp, batchXB, upName, batchSize, cfg.HiddenDim, cfg.Dim) {
				for b := 0; b < batchSize; b++ {
					w.MatMul(e.GEMV, batchUp[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], batchXB[b*cfg.Dim:(b+1)*cfg.Dim], upName, cfg.HiddenDim, cfg.Dim)
				}
			}
			if isMetal {
				metal.SwiGLU(batchGate, batchUp, batchSize*cfg.HiddenDim)
			} else {
				for b := 0; b < batchSize; b++ {
					math.SwiGLU(batchGate[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], batchUp[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], cfg.HiddenDim)
				}
			}
		}

		// Batched FFN Down & Residual
		if !w.MatMulBatch(batchFFNDown, batchGate, downName, batchSize, cfg.Dim, cfg.HiddenDim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchFFNDown[b*cfg.Dim:(b+1)*cfg.Dim], batchGate[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], downName, cfg.Dim, cfg.HiddenDim)
			}
		}
		if isMetal {
			metal.AddResidual(batchX, batchFFNDown, batchSize*cfg.Dim)
			metal.EndBatch()
		} else {
			for i := range batchX {
				batchX[i] += batchFFNDown[i]
			}
		}
	}

	// Final Norm & Logits for the last token in batch
	lastX := batchX[(batchSize-1)*cfg.Dim : batchSize*cfg.Dim]
	outputNormW := w.Get1DWeight("output_norm.weight", cfg.Dim)
	if !isMetal || metal.RMSNorm(e.Arena.XB, lastX, outputNormW, cfg.Dim, cfg.Eps) != nil {
		math.RMSNorm(e.Arena.XB, lastX, outputNormW, cfg.Eps)
	}

	w.MatMul(e.GEMV, e.Arena.Logits, e.Arena.XB, "output.weight", cfg.VocabSize, cfg.Dim)
	return e.Arena.Logits
}

// ForwardLayerRange executes a specific range of transformer layers [startLayer, endLayer] on activation x.
func (e *Engine) ForwardLayerRange(x []float32, startLayer, endLayer int, pos int, kv *KVCache) []float32 {
	cfg := e.Config
	headDim := cfg.HeadDim()
	kvDim := cfg.KVDim()
	kvMul := cfg.KVMul()
	a := e.Arena
	w := e.Weights

	if startLayer < 0 {
		startLayer = 0
	}
	if endLayer >= cfg.NumLayers {
		endLayer = cfg.NumLayers - 1
	}

	copy(a.X, x)
	activeContext := pos + 1
	if activeContext > kv.MaxSeq {
		activeContext = kv.MaxSeq
	}
	attnScale := float32(1.0 / math_sqrt(float64(headDim)))

	for l := startLayer; l <= endLayer; l++ {
		attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), cfg.Dim)
		math.RMSNorm(a.XB, a.X, attnNormW, cfg.Eps)

		w.MatMul(e.GEMV, a.Q, a.XB, fmt.Sprintf("blk.%d.attn_q.weight", l), cfg.Dim, cfg.Dim)
		w.MatMul(e.GEMV, a.K, a.XB, fmt.Sprintf("blk.%d.attn_k.weight", l), kvDim, cfg.Dim)
		w.MatMul(e.GEMV, a.V, a.XB, fmt.Sprintf("blk.%d.attn_v.weight", l), kvDim, cfg.Dim)

		for h := 0; h < cfg.NumHeads; h++ {
			math.ApplyRoPE(a.Q[h*headDim:(h+1)*headDim], pos, headDim, cfg.RopeTheta)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			math.ApplyRoPE(a.K[h*headDim:(h+1)*headDim], pos, headDim, cfg.RopeTheta)
		}

		slot := pos % kv.MaxSeq
		cacheOffset := slot * kvDim
		copy(kv.Key[l][cacheOffset:cacheOffset+kvDim], a.K)
		copy(kv.Value[l][cacheOffset:cacheOffset+kvDim], a.V)

		for i := range a.AttnOut {
			a.AttnOut[i] = 0
		}

		for h := 0; h < cfg.NumHeads; h++ {
			qHead := a.Q[h*headDim : (h+1)*headDim]
			kvHeadIdx := h / kvMul
			scores := a.AttnScores[:activeContext]

			for t := 0; t < activeContext; t++ {
				kHead := kv.Key[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qHead[d] * kHead[d]
				}
				scores[t] = dot * attnScale
			}

			math.Softmax(scores)

			outHead := a.AttnOut[h*headDim : (h+1)*headDim]
			for t := 0; t < activeContext; t++ {
				weight := scores[t]
				vHead := kv.Value[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
				for d := 0; d < headDim; d++ {
					outHead[d] += weight * vHead[d]
				}
			}
		}

		w.MatMul(e.GEMV, a.AttnProj, a.AttnOut, fmt.Sprintf("blk.%d.attn_output.weight", l), cfg.Dim, cfg.Dim)
		for i := 0; i < cfg.Dim; i++ {
			a.X[i] += a.AttnProj[i]
		}

		ffnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.ffn_norm.weight", l), cfg.Dim)
		math.RMSNorm(a.XB, a.X, ffnNormW, cfg.Eps)

		w.MatMul(e.GEMV, a.Gate, a.XB, fmt.Sprintf("blk.%d.ffn_gate.weight", l), cfg.HiddenDim, cfg.Dim)
		w.MatMul(e.GEMV, a.Up, a.XB, fmt.Sprintf("blk.%d.ffn_up.weight", l), cfg.HiddenDim, cfg.Dim)
		math.SwiGLU(a.Gate, a.Up, cfg.HiddenDim)

		w.MatMul(e.GEMV, a.FFNDown, a.Gate, fmt.Sprintf("blk.%d.ffn_down.weight", l), cfg.Dim, cfg.HiddenDim)
		for i := 0; i < cfg.Dim; i++ {
			a.X[i] += a.FFNDown[i]
		}
	}

	res := make([]float32, cfg.Dim)
	copy(res, a.X)
	return res
}

// ForwardLogits computes the final output norm and logits from an activation vector x.
func (e *Engine) ForwardLogits(x []float32) []float32 {
	cfg := e.Config
	a := e.Arena
	w := e.Weights
	copy(a.X, x)

	outputNormW := w.Get1DWeight("output_norm.weight", cfg.Dim)
	math.RMSNorm(a.XB, a.X, outputNormW, cfg.Eps)

	w.MatMul(e.GEMV, a.Logits, a.XB, "output.weight", cfg.VocabSize, cfg.Dim)
	res := make([]float32, cfg.VocabSize)
	copy(res, a.Logits)
	return res
}

func math_sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func (e *Engine) forwardQwen35SSMLayer(l int, a *MemoryArena, kv *KVCache, cfg ModelConfig) {
	w := e.Weights
	ssmInner := cfg.SSMInnerSize                   // 6144
	ssmState := cfg.SSMStateSize                   // 128
	ssmGroups := cfg.SSMGroupCount                 // 16
	ssmRank := cfg.SSMTimeStepRank                 // 48
	ssmChannels := ssmInner + 2*ssmGroups*ssmState // 10240

	// 1. Pre-SSM RMSNorm
	attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), cfg.Dim)
	math.RMSNorm(a.XB, a.X, attnNormW, cfg.Eps)

	// 2. Project gate (6144) and qkv (10240)
	w.MatMul(e.GEMV, a.SSMGate, a.XB, fmt.Sprintf("blk.%d.attn_gate.weight", l), ssmInner, cfg.Dim)
	w.MatMul(e.GEMV, a.SSMQKV, a.XB, fmt.Sprintf("blk.%d.attn_qkv.weight", l), ssmChannels, cfg.Dim)

	// 3. 1D Depthwise Convolution
	convWeight := w.Get1DWeight(fmt.Sprintf("blk.%d.ssm_conv1d.weight", l), cfg.SSMConvKernel*ssmChannels)
	if kv.SSM != nil && l < len(kv.SSM.ConvState) {
		math.Conv1DStep(a.SSMConvOut, a.SSMQKV, kv.SSM.ConvState[l], convWeight, cfg.SSMConvKernel, ssmChannels)
	} else {
		copy(a.SSMConvOut, a.SSMQKV)
		math.SiLU(a.SSMConvOut)
	}

	// 4. Time-step Delta & decay parameters
	w.MatMul(e.GEMV, a.SSMAlpha, a.XB, fmt.Sprintf("blk.%d.ssm_alpha.weight", l), ssmRank, cfg.Dim)
	w.MatMul(e.GEMV, a.SSMBeta, a.XB, fmt.Sprintf("blk.%d.ssm_beta.weight", l), ssmRank, cfg.Dim)
	dtBias := w.Get1DBias(fmt.Sprintf("blk.%d.ssm_dt.bias", l))
	ssmA := w.Get1DBias(fmt.Sprintf("blk.%d.ssm_a", l))

	// Softplus on dt: dt = log(1 + exp(alpha + dt_bias))
	dt := make([]float32, ssmRank)
	for i := 0; i < ssmRank; i++ {
		val := float64(a.SSMAlpha[i])
		if dtBias != nil && i < len(dtBias) {
			val += float64(dtBias[i])
		}
		if val > 20.0 {
			dt[i] = float32(val)
		} else {
			dt[i] = float32(stdmath.Log(1.0 + stdmath.Exp(val)))
		}
	}

	// 5. DeltaNet / Recurrent State Update
	// Split conv output:
	// Q: first 16 * 128 = 2048
	// K: next 16 * 128 = 2048
	// V: remaining 48 * 128 = 6144
	qBase := 0
	kBase := ssmGroups * ssmState      // 2048
	vBase := 2 * ssmGroups * ssmState  // 4096

	qVec := a.SSMConvOut[qBase : qBase+ssmGroups*ssmState]
	kVec := a.SSMConvOut[kBase : kBase+ssmGroups*ssmState]
	vVec := a.SSMConvOut[vBase : vBase+ssmInner]

	numHeads := ssmRank // 48

	ssmNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.ssm_norm.weight", l), ssmState)

	for h := 0; h < numHeads; h++ {
		g := h % ssmGroups // Group index in [0, 16) mapped modulo ssmGroups (matches llama.cpp)
		qHead := qVec[g*ssmState : (g+1)*ssmState]
		kHead := kVec[g*ssmState : (g+1)*ssmState]
		vHead := vVec[h*ssmState : (h+1)*ssmState]
		outHead := a.SSMOut[h*ssmState : (h+1)*ssmState]

		// Q L2-norm with 1/sqrt(ssm_state) scaling; K L2-norm
		var qSq float32 = 0
		var kSq float32 = 0
		for i := 0; i < ssmState; i++ {
			qSq += qHead[i] * qHead[i]
			kSq += kHead[i] * kHead[i]
		}
		invQ := float32(1.0 / (stdmath.Sqrt(float64(qSq)+1e-6) * stdmath.Sqrt(float64(ssmState))))
		invK := float32(1.0 / stdmath.Sqrt(float64(kSq)+1e-6))

		// Decay factor = exp(dt * A)
		decay := float32(1.0)
		if ssmA != nil && h < len(ssmA) {
			decay = float32(stdmath.Exp(float64(dt[h] * ssmA[h])))
		}

		betaVal := float32(1.0 / (1.0 + stdmath.Exp(float64(-a.SSMBeta[h]))))

		if kv.SSM != nil && l < len(kv.SSM.SSMState) {
			stateHead := kv.SSM.SSMState[l][h*ssmState*ssmState : (h+1)*ssmState*ssmState]
			for i := 0; i < ssmState; i++ {
				sRow := stateHead[i*ssmState : (i+1)*ssmState]
				// 1. Decay state first (matches upstream llama.cpp)
				for j := 0; j < ssmState; j++ {
					sRow[j] *= decay
				}
				// 2. Memory retrieval on decayed state
				var kvMem float32 = 0
				for j := 0; j < ssmState; j++ {
					kvMem += sRow[j] * (kHead[j] * invK)
				}
				// 3. Error delta
				delta := (vHead[i] - kvMem) * betaVal
				// 4. Update state and output projection
				var yVal float32 = 0
				for j := 0; j < ssmState; j++ {
					kj := kHead[j] * invK
					qj := qHead[j] * invQ
					s := sRow[j] + delta*kj
					sRow[j] = s
					yVal += s * qj
				}
				outHead[i] = yVal
			}
		} else {
			// Stateless fallback
			dot := float32(0)
			for j := 0; j < ssmState; j++ {
				dot += (kHead[j] * invK) * (qHead[j] * invQ)
			}
			for i := 0; i < ssmState; i++ {
				outHead[i] = vHead[i] * betaVal * dot
			}
		}

		// Apply ssm_norm on head output
		math.RMSNorm(outHead, outHead, ssmNormW, cfg.Eps)

		// Gate with SiLU(gate)
		for i := 0; i < ssmState; i++ {
			gIdx := h*ssmState + i
			gVal := a.SSMGate[gIdx]
			siluGate := gVal / (1.0 + float32(stdmath.Exp(float64(-gVal))))
			outHead[i] *= siluGate
		}
	}

	// 6. SSM Output projection back to Dim (6144 -> 5120)
	w.MatMul(e.GEMV, a.AttnProj, a.SSMOut, fmt.Sprintf("blk.%d.ssm_out.weight", l), cfg.Dim, ssmInner)
	for i := 0; i < cfg.Dim; i++ {
		a.X[i] += a.AttnProj[i]
	}

	// 7. Feed-Forward (SwiGLU MLP)
	postNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.post_attention_norm.weight", l), cfg.Dim)
	math.RMSNorm(a.XB, a.X, postNormW, cfg.Eps)

	w.MatMul(e.GEMV, a.Gate, a.XB, fmt.Sprintf("blk.%d.ffn_gate.weight", l), cfg.HiddenDim, cfg.Dim)
	w.MatMul(e.GEMV, a.Up, a.XB, fmt.Sprintf("blk.%d.ffn_up.weight", l), cfg.HiddenDim, cfg.Dim)
	math.SwiGLU(a.Gate, a.Up, cfg.HiddenDim)

	w.MatMul(e.GEMV, a.FFNDown, a.Gate, fmt.Sprintf("blk.%d.ffn_down.weight", l), cfg.Dim, cfg.HiddenDim)
	for i := 0; i < cfg.Dim; i++ {
		a.X[i] += a.FFNDown[i]
	}
}

func (e *Engine) forwardQwen35AttentionLayer(l int, pos int, activeContext int, a *MemoryArena, kv *KVCache, cfg ModelConfig) {
	w := e.Weights
	headDim := cfg.HeadDim() // 256
	kvDim := cfg.KVDim()     // 1024
	attnDim := cfg.AttnDim() // 6144
	kvMul := cfg.KVMul()     // 6
	attnScale := float32(1.0 / math_sqrt(float64(headDim)))
	ropeDim := cfg.RopeDim
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}

	// 1. Pre-Attention RMSNorm
	attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), cfg.Dim)
	math.RMSNorm(a.XB, a.X, attnNormW, cfg.Eps)

	// 2. Q, K, V Projections
	// Q projection has gate concatenated: rows = 12288 (24 heads x (256 Q + 256 Gate))
	w.MatMul(e.GEMV, a.QRaw, a.XB, fmt.Sprintf("blk.%d.attn_q.weight", l), 12288, cfg.Dim)
	for h := 0; h < cfg.NumHeads; h++ {
		copy(a.Q[h*headDim:(h+1)*headDim], a.QRaw[h*512:h*512+256])
		copy(a.AttnGate[h*headDim:(h+1)*headDim], a.QRaw[h*512+256:(h+1)*512])
	}
	w.MatMul(e.GEMV, a.K, a.XB, fmt.Sprintf("blk.%d.attn_k.weight", l), kvDim, cfg.Dim)
	w.MatMul(e.GEMV, a.V, a.XB, fmt.Sprintf("blk.%d.attn_v.weight", l), kvDim, cfg.Dim)

	// Per-head Q and K normalization
	qNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_q_norm.weight", l), headDim)
	kNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_k_norm.weight", l), headDim)

	for h := 0; h < cfg.NumHeads; h++ {
		qHead := a.Q[h*headDim : (h+1)*headDim]
		math.RMSNorm(qHead, qHead, qNormW, cfg.Eps)
		math.ApplyRoPE(qHead[:ropeDim], pos, ropeDim, cfg.RopeTheta)
	}

	for h := 0; h < cfg.NumKVHeads; h++ {
		kHead := a.K[h*headDim : (h+1)*headDim]
		math.RMSNorm(kHead, kHead, kNormW, cfg.Eps)
		math.ApplyRoPE(kHead[:ropeDim], pos, ropeDim, cfg.RopeTheta)
	}

	// Store into KV cache
	slot := pos % kv.MaxSeq
	cacheOffset := slot * kvDim
	copy(kv.Key[l][cacheOffset:cacheOffset+kvDim], a.K[:kvDim])
	copy(kv.Value[l][cacheOffset:cacheOffset+kvDim], a.V[:kvDim])

	// Multi-Head Attention with GQA
	for i := 0; i < attnDim; i++ {
		a.AttnOut[i] = 0
	}

	for h := 0; h < cfg.NumHeads; h++ {
		qHead := a.Q[h*headDim : (h+1)*headDim]
		kvHeadIdx := h / kvMul
		scores := a.AttnScores[:activeContext]

		for t := 0; t < activeContext; t++ {
			kHead := kv.Key[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
			scores[t] = quant.DotVecF32(qHead, kHead) * attnScale
		}

		math.Softmax(scores)

		outHead := a.AttnOut[h*headDim : (h+1)*headDim]
		for t := 0; t < activeContext; t++ {
			vHead := kv.Value[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
			weight := scores[t]
			for d := 0; d < headDim; d++ {
				outHead[d] += weight * vHead[d]
			}
		}
	}

	// Post-attention output gating: AttnOut[i] *= sigmoid(AttnGate[i])
	for i := 0; i < attnDim; i++ {
		gVal := a.AttnGate[i]
		sig := 1.0 / (1.0 + float32(stdmath.Exp(float64(-gVal))))
		a.AttnOut[i] *= sig
	}

	// 3. Output projection back to Dim (6144 -> 5120)
	w.MatMul(e.GEMV, a.AttnProj, a.AttnOut, fmt.Sprintf("blk.%d.attn_output.weight", l), cfg.Dim, attnDim)
	for i := 0; i < cfg.Dim; i++ {
		a.X[i] += a.AttnProj[i]
	}

	// 4. Feed-Forward (SwiGLU MLP)
	postNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.post_attention_norm.weight", l), cfg.Dim)
	math.RMSNorm(a.XB, a.X, postNormW, cfg.Eps)

	w.MatMul(e.GEMV, a.Gate, a.XB, fmt.Sprintf("blk.%d.ffn_gate.weight", l), cfg.HiddenDim, cfg.Dim)
	w.MatMul(e.GEMV, a.Up, a.XB, fmt.Sprintf("blk.%d.ffn_up.weight", l), cfg.HiddenDim, cfg.Dim)
	math.SwiGLU(a.Gate, a.Up, cfg.HiddenDim)

	w.MatMul(e.GEMV, a.FFNDown, a.Gate, fmt.Sprintf("blk.%d.ffn_down.weight", l), cfg.Dim, cfg.HiddenDim)
	for i := 0; i < cfg.Dim; i++ {
		a.X[i] += a.FFNDown[i]
	}
}

func (e *Engine) matMulBatch(y, x []float32, tensorName string, batchSize, rows, cols int) {
	if !e.Weights.MatMulBatch(y, x, tensorName, batchSize, rows, cols) {
		for b := 0; b < batchSize; b++ {
			e.Weights.MatMul(e.GEMV, y[b*rows:(b+1)*rows], x[b*cols:(b+1)*cols], tensorName, rows, cols)
		}
	}
}

func (e *Engine) forwardBatchQwen35(tokens []int, startPos int, kv *KVCache, cfg ModelConfig) []float32 {
	batchSize := len(tokens)
	w := e.Weights
	dim := cfg.Dim
	hiddenDim := cfg.HiddenDim
	ssmInner := cfg.SSMInnerSize                   // 6144
	ssmState := cfg.SSMStateSize                   // 128
	ssmGroups := cfg.SSMGroupCount                 // 16
	ssmRank := cfg.SSMTimeStepRank                 // 48
	ssmChannels := ssmInner + 2*ssmGroups*ssmState // 10240
	headDim := cfg.HeadDim()                       // 256
	kvDim := cfg.KVDim()                           // 1024
	attnDim := cfg.AttnDim()                       // 6144
	kvMul := cfg.KVMul()
	attnScale := float32(1.0 / math_sqrt(float64(headDim)))
	ropeDim := cfg.RopeDim
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}

	// 1. Embedding matrix (batchSize x Dim)
	batchX := make([]float32, batchSize*dim)
	if metal.IsAvailable() && e.TokenEmbdBuf != nil && e.TokenEmbdTyp == int(gguf.GGMLTypeQ4_K) {
		if err := metal.EmbedLookupQ4KBatch(batchX, e.TokenEmbdBuf, tokens, dim); err != nil {
			for i, tok := range tokens {
				w.ExtractEmbedding(tok, batchX[i*dim:(i+1)*dim], dim)
			}
		}
	} else {
		for i, tok := range tokens {
			w.ExtractEmbedding(tok, batchX[i*dim:(i+1)*dim], dim)
		}
	}

	batchXB := make([]float32, batchSize*dim)

	// SSM Layer Buffers
	batchSSMGate := make([]float32, batchSize*ssmInner)
	batchSSMQKV := make([]float32, batchSize*ssmChannels)
	batchSSMConvOut := make([]float32, batchSize*ssmChannels)
	batchSSMAlpha := make([]float32, batchSize*ssmRank)
	batchSSMBeta := make([]float32, batchSize*ssmRank)
	batchSSMOut := make([]float32, batchSize*ssmInner)

	// Attention Layer Buffers
	batchQRaw := make([]float32, batchSize*12288)
	batchQ := make([]float32, batchSize*attnDim)
	batchAttnGate := make([]float32, batchSize*attnDim)
	batchK := make([]float32, batchSize*kvDim)
	batchV := make([]float32, batchSize*kvDim)
	batchAttnOut := make([]float32, batchSize*attnDim)

	// Common Layer Buffers
	batchAttnProj := make([]float32, batchSize*dim)
	batchGate := make([]float32, batchSize*hiddenDim)
	var batchUp []float32
	batchFFNDown := make([]float32, batchSize*dim)


	// Process all layers
	for l := 0; l < cfg.NumLayers; l++ {
		// 1. Pre-norm RMSNorm
		attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), dim)
		if !metal.IsAvailable() || metal.RMSNormBatch(batchXB, batchX, attnNormW, dim, cfg.Eps, batchSize) != nil {
			for b := 0; b < batchSize; b++ {
				math.RMSNorm(batchXB[b*dim:(b+1)*dim], batchX[b*dim:(b+1)*dim], attnNormW, cfg.Eps)
			}
		}

		if cfg.IsSSMLayer(l) {
			// SSM Layer
			// A. Projections
			if metal.IsAvailable() {
				metal.BeginBatch()
			}
			e.matMulBatch(batchSSMGate, batchXB, fmt.Sprintf("blk.%d.attn_gate.weight", l), batchSize, ssmInner, dim)
			e.matMulBatch(batchSSMQKV, batchXB, fmt.Sprintf("blk.%d.attn_qkv.weight", l), batchSize, ssmChannels, dim)
			e.matMulBatch(batchSSMAlpha, batchXB, fmt.Sprintf("blk.%d.ssm_alpha.weight", l), batchSize, ssmRank, dim)
			e.matMulBatch(batchSSMBeta, batchXB, fmt.Sprintf("blk.%d.ssm_beta.weight", l), batchSize, ssmRank, dim)
			if metal.IsAvailable() {
				metal.EndBatch()
			}

			// B. 1D Causal Convolution
			convWeight := w.Get1DWeight(fmt.Sprintf("blk.%d.ssm_conv1d.weight", l), cfg.SSMConvKernel*ssmChannels)
			var convState []float32
			if kv.SSM != nil && l < len(kv.SSM.ConvState) {
				convState = kv.SSM.ConvState[l]
			}

			if metal.IsAvailable() && len(convWeight) > 0 {
				_ = metal.Conv1DBatch(batchSSMConvOut, batchSSMQKV, convState, convWeight, cfg.SSMConvKernel, ssmChannels, batchSize)
			} else {
				// Run causal conv across all tokens in the batch
				for b := 0; b < batchSize; b++ {
					u := batchSSMQKV[b*ssmChannels : (b+1)*ssmChannels]
					out := batchSSMConvOut[b*ssmChannels : (b+1)*ssmChannels]
					if convState != nil {
						math.Conv1DStep(out, u, convState, convWeight, cfg.SSMConvKernel, ssmChannels)
					} else {
						copy(out, u)
						math.SiLU(out)
					}
				}
			}

			// C. DeltaNet SSM Recurrence
			dtBias := w.Get1DBias(fmt.Sprintf("blk.%d.ssm_dt.bias", l))
			ssmA := w.Get1DBias(fmt.Sprintf("blk.%d.ssm_a", l))
			ssmNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.ssm_norm.weight", l), ssmState)

			var ssmStateSlice []float32
			if kv.SSM != nil && l < len(kv.SSM.SSMState) {
				ssmStateSlice = kv.SSM.SSMState[l]
			}

			if metal.IsAvailable() && len(ssmStateSlice) > 0 {
				_ = metal.SSMRecurrenceBatch(
					batchSSMOut, batchSSMConvOut, batchSSMAlpha, batchSSMBeta,
					dtBias, ssmA, ssmNormW, batchSSMGate, ssmStateSlice,
					ssmInner, ssmState, ssmGroups, ssmRank, cfg.Eps, batchSize, ssmChannels,
				)
			} else {
				// Parallelized DeltaNet SSM Recurrence across heads
				numWorkers := 8
				headsPerWorker := (ssmRank + numWorkers - 1) / numWorkers
				var wg sync.WaitGroup

				for wId := 0; wId < numWorkers; wId++ {
					startH := wId * headsPerWorker
					endH := startH + headsPerWorker
					if endH > ssmRank {
						endH = ssmRank
					}
					if startH >= endH {
						continue
					}

					wg.Add(1)
					go func(sH, eH int) {
						defer wg.Done()
						localDt := make([]float32, ssmRank)
						kScaled := make([]float32, ssmState)
						qScaled := make([]float32, ssmState)

						for b := 0; b < batchSize; b++ {
							alpha := batchSSMAlpha[b*ssmRank : (b+1)*ssmRank]
							beta := batchSSMBeta[b*ssmRank : (b+1)*ssmRank]
							gate := batchSSMGate[b*ssmInner : (b+1)*ssmInner]
							convOut := batchSSMConvOut[b*ssmChannels : (b+1)*ssmChannels]
							ssmOut := batchSSMOut[b*ssmInner : (b+1)*ssmInner]

							for h := sH; h < eH; h++ {
								val := float64(alpha[h])
								if dtBias != nil && h < len(dtBias) {
									val += float64(dtBias[h])
								}
								if val > 20.0 {
									localDt[h] = float32(val)
								} else {
									localDt[h] = float32(stdmath.Log(1.0 + stdmath.Exp(val)))
								}

								g := h % ssmGroups
								qBase := 0
								kBase := ssmGroups * ssmState
								vBase := 2 * ssmGroups * ssmState

								qHead := convOut[qBase+g*ssmState : qBase+(g+1)*ssmState]
								kHead := convOut[kBase+g*ssmState : kBase+(g+1)*ssmState]
								vHead := convOut[vBase+h*ssmState : vBase+(h+1)*ssmState]
								outHead := ssmOut[h*ssmState : (h+1)*ssmState]

								var qSq float32 = 0
								var kSq float32 = 0
								for i := 0; i < ssmState; i++ {
									qSq += qHead[i] * qHead[i]
									kSq += kHead[i] * kHead[i]
								}
								invQ := float32(1.0 / (stdmath.Sqrt(float64(qSq)+1e-6) * stdmath.Sqrt(float64(ssmState))))
								invK := float32(1.0 / stdmath.Sqrt(float64(kSq)+1e-6))

								for i := 0; i < ssmState; i++ {
									kScaled[i] = kHead[i] * invK
									qScaled[i] = qHead[i] * invQ
								}

								decay := float32(1.0)
								if ssmA != nil && h < len(ssmA) {
									decay = float32(stdmath.Exp(float64(localDt[h] * ssmA[h])))
								}
								betaVal := float32(1.0 / (1.0 + stdmath.Exp(float64(-beta[h]))))

								if kv.SSM != nil && l < len(kv.SSM.SSMState) {
									stateHead := kv.SSM.SSMState[l][h*ssmState*ssmState : (h+1)*ssmState*ssmState]
									for i := 0; i < ssmState; i++ {
										sRow := stateHead[i*ssmState : (i+1)*ssmState]
										var kvMem float32 = 0
										for j := 0; j < ssmState; j++ {
											s := sRow[j] * decay
											sRow[j] = s
											kvMem += s * kScaled[j]
										}
										delta := (vHead[i] - kvMem) * betaVal
										var yVal float32 = 0
										for j := 0; j < ssmState; j++ {
											s := sRow[j] + delta*kScaled[j]
											sRow[j] = s
											yVal += s * qScaled[j]
										}
										outHead[i] = yVal
									}
								}

								math.RMSNorm(outHead, outHead, ssmNormW, cfg.Eps)
								for i := 0; i < ssmState; i++ {
									gVal := gate[h*ssmState+i]
									siluGate := gVal / (1.0 + float32(stdmath.Exp(float64(-gVal))))
									outHead[i] *= siluGate
								}
							}
						}
					}(startH, endH)
				}
				wg.Wait()
			}

			// D. Output Projection back to Dim
			e.matMulBatch(batchAttnProj, batchSSMOut, fmt.Sprintf("blk.%d.ssm_out.weight", l), batchSize, dim, ssmInner)
		} else {
			// Attention Layer
			// A. Projections
			if metal.IsAvailable() {
				metal.BeginBatch()
			}
			e.matMulBatch(batchQRaw, batchXB, fmt.Sprintf("blk.%d.attn_q.weight", l), batchSize, 12288, dim)
			e.matMulBatch(batchK, batchXB, fmt.Sprintf("blk.%d.attn_k.weight", l), batchSize, kvDim, dim)
			e.matMulBatch(batchV, batchXB, fmt.Sprintf("blk.%d.attn_v.weight", l), batchSize, kvDim, dim)
			if metal.IsAvailable() {
				metal.EndBatch()
			}

			qNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_q_norm.weight", l), headDim)
			kNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_k_norm.weight", l), headDim)

			if metal.IsAvailable() {
				_ = metal.SplitQGateBatch(batchQ, batchAttnGate, batchQRaw, batchSize)
				_ = metal.RoPENormBatch(batchQ, batchK, qNormW, kNormW, startPos, cfg.NumHeads, cfg.NumKVHeads, headDim, ropeDim, cfg.RopeTheta, cfg.Eps, batchSize)
				_ = metal.KVWriteBatch(kv.Key[l], kv.Value[l], batchK, batchV, startPos, kv.MaxSeq, kvDim, batchSize)
				_ = metal.AttentionGQABatch(batchAttnOut, batchQ, kv.Key[l], kv.Value[l], cfg.NumHeads, cfg.NumKVHeads, headDim, startPos, kv.MaxSeq, attnScale, batchSize)
				_ = metal.AttnGate(batchAttnOut, batchAttnGate, batchSize*attnDim)
			} else {
				for b := 0; b < batchSize; b++ {
					pos := startPos + b
					qRawB := batchQRaw[b*12288 : (b+1)*12288]
					qB := batchQ[b*attnDim : (b+1)*attnDim]
					attnGateB := batchAttnGate[b*attnDim : (b+1)*attnDim]
					kB := batchK[b*kvDim : (b+1)*kvDim]
					vB := batchV[b*kvDim : (b+1)*kvDim]
					attnOutB := batchAttnOut[b*attnDim : (b+1)*attnDim]

					// Deinterleave Q and Gate
					for h := 0; h < cfg.NumHeads; h++ {
						copy(qB[h*headDim:(h+1)*headDim], qRawB[h*512:h*512+256])
						copy(attnGateB[h*headDim:(h+1)*headDim], qRawB[h*512+256:(h+1)*512])
					}

					// Q and K norm + RoPE
					for h := 0; h < cfg.NumHeads; h++ {
						qh := qB[h*headDim : (h+1)*headDim]
						math.RMSNorm(qh, qh, qNormW, cfg.Eps)
						math.ApplyRoPE(qh[:ropeDim], pos, ropeDim, cfg.RopeTheta)
					}
					for h := 0; h < cfg.NumKVHeads; h++ {
						kh := kB[h*headDim : (h+1)*headDim]
						math.RMSNorm(kh, kh, kNormW, cfg.Eps)
						math.ApplyRoPE(kh[:ropeDim], pos, ropeDim, cfg.RopeTheta)
					}

					// Write into KV cache
					slot := pos % kv.MaxSeq
					cacheOffset := slot * kvDim
					copy(kv.Key[l][cacheOffset:cacheOffset+kvDim], kB[:kvDim])
					copy(kv.Value[l][cacheOffset:cacheOffset+kvDim], vB[:kvDim])

					// Causal Attention
					activeContext := pos + 1
					if activeContext > kv.MaxSeq {
						activeContext = kv.MaxSeq
					}
					for i := 0; i < attnDim; i++ {
						attnOutB[i] = 0
					}

					numAttnWorkers := 4
					headsPerWorker := (cfg.NumHeads + numAttnWorkers - 1) / numAttnWorkers
					var attnWg sync.WaitGroup
					for wId := 0; wId < numAttnWorkers; wId++ {
						sH := wId * headsPerWorker
						eH := sH + headsPerWorker
						if eH > cfg.NumHeads {
							eH = cfg.NumHeads
						}
						if sH >= eH {
							continue
						}
						attnWg.Add(1)
						go func(startH, endH int) {
							defer attnWg.Done()
							scores := make([]float32, activeContext)
							for h := startH; h < endH; h++ {
								qh := qB[h*headDim : (h+1)*headDim]
								kvHeadIdx := h / kvMul
								for t := 0; t < activeContext; t++ {
									kHead := kv.Key[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
									scores[t] = quant.DotVecF32(qh, kHead) * attnScale
								}
								math.Softmax(scores)

								outHead := attnOutB[h*headDim : (h+1)*headDim]
								for t := 0; t < activeContext; t++ {
									vHead := kv.Value[l][t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
									wVal := scores[t]
									for d := 0; d < headDim; d++ {
										outHead[d] += wVal * vHead[d]
									}
								}
							}
						}(sH, eH)
					}
					attnWg.Wait()

					// Post-attention output gating
					for i := 0; i < attnDim; i++ {
						gVal := attnGateB[i]
						sig := float32(1.0 / (1.0 + stdmath.Exp(float64(-gVal))))
						attnOutB[i] *= sig
					}
				}
			}

			// Output projection back to Dim
			e.matMulBatch(batchAttnProj, batchAttnOut, fmt.Sprintf("blk.%d.attn_output.weight", l), batchSize, dim, attnDim)
		}

		// Residual Add & FFN Norm (fused when GPU is available)
		postNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.post_attention_norm.weight", l), dim)
		if !metal.IsAvailable() || metal.ResidualRMSNormBatch(batchX, batchAttnProj, batchXB, postNormW, dim, cfg.Eps, batchSize) != nil {
			if metal.IsAvailable() {
				metal.AddResidual(batchX, batchAttnProj, batchSize*dim)
			} else {
				for i := 0; i < batchSize*dim; i++ {
					batchX[i] += batchAttnProj[i]
				}
			}
			for b := 0; b < batchSize; b++ {
				math.RMSNorm(batchXB[b*dim:(b+1)*dim], batchX[b*dim:(b+1)*dim], postNormW, cfg.Eps)
			}
		}

		gateName := fmt.Sprintf("blk.%d.ffn_gate.weight", l)
		upName := fmt.Sprintf("blk.%d.ffn_up.weight", l)
		downName := fmt.Sprintf("blk.%d.ffn_down.weight", l)

		if metal.IsAvailable() {
			metal.BeginBatch()
			if batchUp == nil {
				batchUp = make([]float32, batchSize*hiddenDim)
			}
			e.matMulBatch(batchGate, batchXB, gateName, batchSize, hiddenDim, dim)
			e.matMulBatch(batchUp, batchXB, upName, batchSize, hiddenDim, dim)
			metal.SwiGLU(batchGate, batchUp, batchSize*hiddenDim)
			e.matMulBatch(batchFFNDown, batchGate, downName, batchSize, dim, hiddenDim)
			metal.AddResidual(batchX, batchFFNDown, batchSize*dim)
			metal.EndBatch()
		} else {
			if batchUp == nil {
				batchUp = make([]float32, batchSize*hiddenDim)
			}
			e.matMulBatch(batchGate, batchXB, gateName, batchSize, hiddenDim, dim)
			e.matMulBatch(batchUp, batchXB, upName, batchSize, hiddenDim, dim)
			for b := 0; b < batchSize; b++ {
				math.SwiGLU(batchGate[b*hiddenDim:(b+1)*hiddenDim], batchUp[b*hiddenDim:(b+1)*hiddenDim], hiddenDim)
			}
			e.matMulBatch(batchFFNDown, batchGate, downName, batchSize, dim, hiddenDim)
			for i := 0; i < batchSize*dim; i++ {
				batchX[i] += batchFFNDown[i]
			}
		}
	}

	// Final RMSNorm on the LAST token
	lastX := batchX[(batchSize-1)*dim : batchSize*dim]
	outputNormW := w.Get1DWeight("output_norm.weight", dim)
	if !metal.IsAvailable() || metal.RMSNorm(e.Arena.XB, lastX, outputNormW, dim, cfg.Eps) != nil {
		math.RMSNorm(e.Arena.XB, lastX, outputNormW, cfg.Eps)
	}

	// Logits projection
	w.MatMul(e.GEMV, e.Arena.Logits, e.Arena.XB, "output.weight", cfg.VocabSize, dim)
	return e.Arena.Logits
}
