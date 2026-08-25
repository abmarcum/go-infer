package engine

import (
	"fmt"
	"go-inference/pkg/gguf"
	"go-inference/pkg/math"
	"go-inference/pkg/metal"
	"go-inference/pkg/quant"
	"unsafe"
)

// Weights manages loaded model tensor weights and dispatching for matrix operations.
type Weights struct {
	F32Weights map[string][]float32
	RawWeights map[string][]byte
	GPUBufs    map[string]unsafe.Pointer
	Meta       map[string]gguf.TensorInfo
}

// NewWeights initializes weights from a GGUF reader.
func NewWeights(reader *gguf.Reader) (*Weights, error) {
	f32Map := make(map[string][]float32)
	rawMap := make(map[string][]byte)
	metaMap := make(map[string]gguf.TensorInfo)
	gpuMap := make(map[string]unsafe.Pointer)

	for name, info := range reader.Header.Tensors {
		metaMap[name] = info
		raw, _, err := reader.GetTensorData(name)
		if err != nil {
			return nil, fmt.Errorf("read tensor %s: %w", name, err)
		}
		rawMap[name] = raw

		// Pre-wrap raw tensor byte slices into persistent GPU buffers if Metal is active
		if metal.IsAvailable() && len(raw) > 0 {
			gpuMap[name] = metal.CreateBuffer(unsafe.Pointer(&raw[0]), len(raw))
		}

		// If it's a 1D normalization weight or bias, cache as F32 for fast direct access
		if len(info.Dimensions) == 1 {
			switch info.Type {
			case gguf.GGMLTypeF32:
				f32Map[name] = quant.DequantizeF32(raw, info.NumElements())
			case gguf.GGMLTypeF16:
				f32Map[name] = quant.DequantizeF16(raw, info.NumElements())
			case gguf.GGMLTypeBF16:
				f32Map[name] = quant.DequantizeBF16(raw, info.NumElements())
			case gguf.GGMLTypeQ4_0:
				f32Map[name] = quant.DequantizeQ4_0(raw, info.NumElements())
			case gguf.GGMLTypeQ4_1:
				f32Map[name] = quant.DequantizeQ4_1(raw, info.NumElements())
			case gguf.GGMLTypeQ5_0:
				f32Map[name] = quant.DequantizeQ5_0(raw, info.NumElements())
			case gguf.GGMLTypeQ5_1:
				f32Map[name] = quant.DequantizeQ5_1(raw, info.NumElements())
			case gguf.GGMLTypeQ8_0:
				f32Map[name] = quant.DequantizeQ8_0(raw, info.NumElements())
			case gguf.GGMLTypeQ4_K:
				f32Map[name] = quant.DequantizeQ4_K(raw, info.NumElements())
			case gguf.GGMLTypeQ6_K:
				f32Map[name] = quant.DequantizeQ6_K(raw, info.NumElements())
			default:
				f32Map[name] = quant.DequantizeF32(raw, info.NumElements())
			}
		} else if info.Type == gguf.GGMLTypeF32 {
			f32Map[name] = quant.DequantizeF32(raw, info.NumElements())
		}
	}

	return &Weights{
		F32Weights: f32Map,
		RawWeights: rawMap,
		GPUBufs:    gpuMap,
		Meta:       metaMap,
	}, nil
}

// ResolveTensorName checks for common GGUF naming variations.
func (w *Weights) ResolveTensorName(name string) (string, bool) {
	if _, ok := w.Meta[name]; ok {
		return name, true
	}

	// Alias mappings for output_norm and token_embd
	aliases := map[string][]string{
		"output_norm.weight": {"norm.weight", "model.norm.weight", "output.norm.weight"},
		"output.weight":      {"token_embd.weight", "model.embed_tokens.weight"},
		"token_embd.weight":  {"model.embed_tokens.weight", "tok_embeddings.weight", "embd.weight"},
	}

	if candidates, ok := aliases[name]; ok {
		for _, cand := range candidates {
			if _, ok := w.Meta[cand]; ok {
				return cand, true
			}
		}
	}

	return name, false
}

// Get1DWeight returns 1D weight tensor or a fallback slice of 1.0s if missing.
func (w *Weights) Get1DWeight(name string, defaultDim int) []float32 {
	resolved, ok := w.ResolveTensorName(name)
	if ok {
		if f32s, exists := w.F32Weights[resolved]; exists && len(f32s) > 0 {
			return f32s
		}
	}

	// Safe fallback: array of 1.0s
	ones := make([]float32, defaultDim)
	for i := range ones {
		ones[i] = 1.0
	}
	return ones
}

// ExtractEmbedding copies the embedding vector for a token into out.
func (w *Weights) ExtractEmbedding(token int, out []float32, dim int) {
	tensorName, ok := w.ResolveTensorName("token_embd.weight")
	if !ok {
		return
	}
	info, ok := w.Meta[tensorName]
	if !ok {
		return
	}
	raw := w.RawWeights[tensorName]

	switch info.Type {
	case gguf.GGMLTypeF32:
		f32s := w.F32Weights[tensorName]
		if f32s == nil {
			f32s = quant.DequantizeF32(raw, info.NumElements())
			w.F32Weights[tensorName] = f32s
		}
		start := token * dim
		if start+dim <= len(f32s) {
			copy(out, f32s[start:start+dim])
		}
	case gguf.GGMLTypeF16:
		byteOffset := token * dim * 2
		if byteOffset+dim*2 <= len(raw) {
			for i := 0; i < dim; i++ {
				h := uint16(raw[byteOffset+i*2]) | (uint16(raw[byteOffset+i*2+1]) << 8)
				out[i] = quant.FP16ToFP32(h)
			}
		}
	case gguf.GGMLTypeBF16:
		byteOffset := token * dim * 2
		if byteOffset+dim*2 <= len(raw) {
			for i := 0; i < dim; i++ {
				h := uint16(raw[byteOffset+i*2]) | (uint16(raw[byteOffset+i*2+1]) << 8)
				out[i] = quant.BF16ToFP32(h)
			}
		}
	case gguf.GGMLTypeQ4_0:
		bytesPerToken := (dim / 32) * 18
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ4_0(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ4_1:
		bytesPerToken := (dim / 32) * 20
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ4_1(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ5_0:
		bytesPerToken := (dim / 32) * 22
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ5_0(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ5_1:
		bytesPerToken := (dim / 32) * 24
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ5_1(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ8_0:
		bytesPerToken := (dim / 32) * 34
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ8_0(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ4_K:
		bytesPerToken := (dim / 256) * 144
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ4_K(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ6_K:
		bytesPerToken := (dim / 256) * 210
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ6_K(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ2_K:
		bytesPerToken := (dim / 256) * 84
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ2_K(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	case gguf.GGMLTypeQ3_K:
		bytesPerToken := (dim / 256) * 110
		byteOffset := token * bytesPerToken
		if byteOffset+bytesPerToken <= len(raw) {
			deq := quant.DequantizeQ3_K(raw[byteOffset:byteOffset+bytesPerToken], dim)
			copy(out, deq)
		}
	}
}

// MatMul executes parallel matrix multiplication y = W * x for the specified tensor.
func (w *Weights) MatMul(gemv *math.GEMVEngine, y, x []float32, tensorName string, rows, cols int) {
	resolvedName, ok := w.ResolveTensorName(tensorName)
	if !ok {
		return
	}

	info, ok := w.Meta[resolvedName]
	if !ok {
		return
	}

	raw := w.RawWeights[resolvedName]

	// 1. Try Apple Metal GPU acceleration if available
	if metal.IsAvailable() {
		if buf, ok := w.GPUBufs[resolvedName]; ok && buf != nil {
			if err := metal.MatMulBuf(int(info.Type), y, x, buf, rows, cols); err == nil {
				return
			}
		}

		switch info.Type {
		case gguf.GGMLTypeQ4_0:
			if err := metal.MatMulQ4_0(y, x, raw, rows, cols); err == nil {
				return
			}
		case gguf.GGMLTypeQ8_0:
			if err := metal.MatMulQ8_0(y, x, raw, rows, cols); err == nil {
				return
			}
		case gguf.GGMLTypeQ4_K:
			if err := metal.MatMulQ4_K(y, x, raw, rows, cols); err == nil {
				return
			}
		case gguf.GGMLTypeQ6_K:
			if err := metal.MatMulQ6_K(y, x, raw, rows, cols); err == nil {
				return
			}
		case gguf.GGMLTypeF16:
			if err := metal.MatMulF16(y, x, raw, rows, cols); err == nil {
				return
			}
		case gguf.GGMLTypeF32:
			f32s := w.F32Weights[resolvedName]
			if f32s == nil {
				f32s = quant.DequantizeF32(raw, rows*cols)
				w.F32Weights[resolvedName] = f32s
			}
			if err := metal.MatMulF32(y, x, f32s, rows, cols); err == nil {
				return
			}
		}
	}

	// 2. Multithreaded CPU Fallback
	switch info.Type {
	case gguf.GGMLTypeQ4_0:
		gemv.MatMulQ4_0(y, x, raw, rows, cols)
	case gguf.GGMLTypeQ8_0:
		gemv.MatMulQ8_0(y, x, raw, rows, cols)
	case gguf.GGMLTypeQ4_K:
		gemv.MatMulQ4_K(y, x, raw, rows, cols)
	case gguf.GGMLTypeQ6_K:
		gemv.MatMulQ6_K(y, x, raw, rows, cols)
	case gguf.GGMLTypeQ2_K:
		gemv.MatMulQ2_K(y, x, raw, rows, cols)
	case gguf.GGMLTypeQ3_K:
		gemv.MatMulQ3_K(y, x, raw, rows, cols)
	case gguf.GGMLTypeF16:
		gemv.MatMulF16(y, x, raw, rows, cols)
	case gguf.GGMLTypeBF16:
		gemv.MatMulBF16(y, x, raw, rows, cols)
	case gguf.GGMLTypeF32:
		f32s := w.F32Weights[resolvedName]
		if f32s == nil {
			f32s = quant.DequantizeF32(raw, rows*cols)
			w.F32Weights[resolvedName] = f32s
		}
		gemv.MatMulF32(y, x, f32s, rows, cols)
	default:
		// Fallback dequantize to F32
		f32s := w.F32Weights[resolvedName]
		if f32s == nil {
			f32s = quant.DequantizeF32(raw, rows*cols)
			w.F32Weights[resolvedName] = f32s
		}
		gemv.MatMulF32(y, x, f32s, rows, cols)
	}
}

// MatMulBatch executes batched matrix multiplication Y (batch x rows) = X (batch x cols) * W^T
func (w *Weights) MatMulBatch(y, x []float32, tensorName string, batchSize, rows, cols int) bool {
	resolvedName, ok := w.ResolveTensorName(tensorName)
	if !ok {
		return false
	}
	info, ok := w.Meta[resolvedName]
	if !ok {
		return false
	}
	raw := w.RawWeights[resolvedName]

	if metal.IsAvailable() {
		if err := metal.MatMulBatch(y, x, raw, info.Type, batchSize, rows, cols); err == nil {
			return true
		}
	}
	return false
}

// Close releases persistent GPU buffer handles and cleans up resources.
func (w *Weights) Close() {
	if w.GPUBufs != nil && metal.IsAvailable() {
		for _, buf := range w.GPUBufs {
			if buf != nil {
				metal.ReleaseBuffer(buf)
			}
		}
		w.GPUBufs = nil
	}
}
