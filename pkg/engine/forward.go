package engine

import (
	"fmt"
	"go-inference/pkg/math"
	"go-inference/pkg/metal"
	"go-inference/pkg/quant"
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

	// 1. Embedding lookup
	w.ExtractEmbedding(token, a.X, cfg.Dim)

	activeContext := pos + 1
	if activeContext > kv.MaxSeq {
		activeContext = kv.MaxSeq
	}
	attnScale := float32(1.0 / math_sqrt(float64(headDim)))

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

// ForwardBatch evaluates a batch of tokens in parallel during prompt prefill.
func (e *Engine) ForwardBatch(tokens []int, kv *KVCache) []float32 {
	batchSize := len(tokens)
	if batchSize == 0 {
		return nil
	}
	if batchSize == 1 {
		return e.Forward(tokens[0], 0, kv)
	}

	cfg := e.Config
	headDim := cfg.HeadDim()
	kvDim := cfg.KVDim()
	kvMul := cfg.KVMul()
	w := e.Weights

	isMetal := metal.IsAvailable()
	if isMetal {
		metal.BeginBatch()
		defer metal.EndBatch()
	}

	// 1. Embedding matrix (batchSize x Dim)
	batchX := make([]float32, batchSize*cfg.Dim)
	for i, tok := range tokens {
		w.ExtractEmbedding(tok, batchX[i*cfg.Dim:(i+1)*cfg.Dim], cfg.Dim)
	}

	batchXB := make([]float32, batchSize*cfg.Dim)
	batchQ := make([]float32, batchSize*cfg.Dim)
	batchK := make([]float32, batchSize*kvDim)
	batchV := make([]float32, batchSize*kvDim)
	batchAttnOut := make([]float32, batchSize*cfg.Dim)
	batchAttnProj := make([]float32, batchSize*cfg.Dim)
	batchGate := make([]float32, batchSize*cfg.HiddenDim)
	batchUp := make([]float32, batchSize*cfg.HiddenDim)
	batchFFNDown := make([]float32, batchSize*cfg.Dim)

	attnScale := float32(1.0 / math_sqrt(float64(headDim)))

	for l := 0; l < cfg.NumLayers; l++ {
		attnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.attn_norm.weight", l), cfg.Dim)
		for b := 0; b < batchSize; b++ {
			math.RMSNorm(batchXB[b*cfg.Dim:(b+1)*cfg.Dim], batchX[b*cfg.Dim:(b+1)*cfg.Dim], attnNormW, cfg.Eps)
		}

		// Batched Q, K, V projections
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

		// Apply RoPE & write into KV cache for each position
		for b := 0; b < batchSize; b++ {
			pos := b
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
			activeContext := b + 1
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

		// Attention Output Projection & Residual
		if !w.MatMulBatch(batchAttnProj, batchAttnOut, fmt.Sprintf("blk.%d.attn_output.weight", l), batchSize, cfg.Dim, cfg.Dim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchAttnProj[b*cfg.Dim:(b+1)*cfg.Dim], batchAttnOut[b*cfg.Dim:(b+1)*cfg.Dim], fmt.Sprintf("blk.%d.attn_output.weight", l), cfg.Dim, cfg.Dim)
			}
		}
		for i := range batchX {
			batchX[i] += batchAttnProj[i]
		}

		// FFN Norm
		ffnNormW := w.Get1DWeight(fmt.Sprintf("blk.%d.ffn_norm.weight", l), cfg.Dim)
		for b := 0; b < batchSize; b++ {
			math.RMSNorm(batchXB[b*cfg.Dim:(b+1)*cfg.Dim], batchX[b*cfg.Dim:(b+1)*cfg.Dim], ffnNormW, cfg.Eps)
		}

		// Batched Gate & Up
		if !w.MatMulBatch(batchGate, batchXB, fmt.Sprintf("blk.%d.ffn_gate.weight", l), batchSize, cfg.HiddenDim, cfg.Dim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchGate[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], batchXB[b*cfg.Dim:(b+1)*cfg.Dim], fmt.Sprintf("blk.%d.ffn_gate.weight", l), cfg.HiddenDim, cfg.Dim)
			}
		}
		if !w.MatMulBatch(batchUp, batchXB, fmt.Sprintf("blk.%d.ffn_up.weight", l), batchSize, cfg.HiddenDim, cfg.Dim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchUp[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], batchXB[b*cfg.Dim:(b+1)*cfg.Dim], fmt.Sprintf("blk.%d.ffn_up.weight", l), cfg.HiddenDim, cfg.Dim)
			}
		}

		// SwiGLU activation
		for b := 0; b < batchSize; b++ {
			math.SwiGLU(batchGate[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], batchUp[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], cfg.HiddenDim)
		}

		// Batched FFN Down & Residual
		if !w.MatMulBatch(batchFFNDown, batchGate, fmt.Sprintf("blk.%d.ffn_down.weight", l), batchSize, cfg.Dim, cfg.HiddenDim) {
			for b := 0; b < batchSize; b++ {
				w.MatMul(e.GEMV, batchFFNDown[b*cfg.Dim:(b+1)*cfg.Dim], batchGate[b*cfg.HiddenDim:(b+1)*cfg.HiddenDim], fmt.Sprintf("blk.%d.ffn_down.weight", l), cfg.Dim, cfg.HiddenDim)
			}
		}
		for i := range batchX {
			batchX[i] += batchFFNDown[i]
		}
	}

	// Final Norm & Logits for the last token in batch
	lastX := batchX[(batchSize-1)*cfg.Dim : batchSize*cfg.Dim]
	outputNormW := w.Get1DWeight("output_norm.weight", cfg.Dim)
	math.RMSNorm(e.Arena.XB, lastX, outputNormW, cfg.Eps)

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
