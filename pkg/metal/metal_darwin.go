//go:build darwin && cgo

package metal

/*
#cgo darwin CFLAGS: -x objective-c -fobjc-arc
#cgo darwin LDFLAGS: -framework Metal -framework Foundation
#include <stdlib.h>
#include "metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

var (
	initOnce       sync.Once
	initErr        error
	errUnsupported = fmt.Errorf("operation unsupported or metal unavailable")
)

// Init initializes the Metal device, command queue, and compiles the MSL kernels.
func Init() error {
	initOnce.Do(func() {
		ret := C.metal_init()
		if ret != 0 {
			initErr = fmt.Errorf("metal_init failed with code %d", int(ret))
		}
	})
	return initErr
}

// IsAvailable returns true if Metal GPU acceleration is initialized and available.
func IsAvailable() bool {
	return bool(C.metal_is_available())
}

// BeginBatch begins recording a single GPU command buffer for a full forward pass.
func BeginBatch() {
	C.metal_begin_batch()
}

// EndBatch commits the recorded GPU command buffer and waits for completion.
func EndBatch() {
	C.metal_end_batch()
}

// CreateBuffer wraps an existing memory slice into a persistent MTLBuffer with zero-copy.
func CreateBuffer(ptr unsafe.Pointer, bytes int) unsafe.Pointer {
	if ptr == nil || bytes == 0 {
		return nil
	}
	return unsafe.Pointer(C.metal_create_buffer(ptr, C.size_t(bytes)))
}

// ReleaseBuffer releases a persistent MTLBuffer reference.
func ReleaseBuffer(buf unsafe.Pointer) {
	if buf != nil {
		C.metal_release_buffer(C.metal_buffer_t(buf))
	}
}

// LayerWeights holds persistent GPU buffer handles for a single transformer layer.
type LayerWeights struct {
	WQBuf, WKBuf, WVBuf, WOBuf          unsafe.Pointer
	WQType, WKType, WVType, WOType      int
	FFNGateBuf, FFNUpBuf, FFNDownBuf    unsafe.Pointer
	FFNGateType, FFNUpType, FFNDownType int
	AttnNormBuf, FFNNormBuf             unsafe.Pointer
}

// PreallocatedLayers stores a C-pinned array of layer weight handles to eliminate runtime allocations.
type PreallocatedLayers struct {
	ptr unsafe.Pointer
	len int
}

// NewPreallocatedLayers pre-packs LayerWeights into a persistent C-allocated buffer.
func NewPreallocatedLayers(layers []LayerWeights) *PreallocatedLayers {
	if len(layers) == 0 {
		return nil
	}
	size := len(layers) * int(unsafe.Sizeof(C.metal_layer_weights_t{}))
	cPtr := C.malloc(C.size_t(size))
	cSlice := (*[1 << 20]C.metal_layer_weights_t)(cPtr)[:len(layers):len(layers)]
	for i, l := range layers {
		cSlice[i].wq = C.metal_buffer_t(l.WQBuf)
		cSlice[i].wq_type = C.int(l.WQType)
		cSlice[i].wk = C.metal_buffer_t(l.WKBuf)
		cSlice[i].wk_type = C.int(l.WKType)
		cSlice[i].wv = C.metal_buffer_t(l.WVBuf)
		cSlice[i].wv_type = C.int(l.WVType)
		cSlice[i].wo = C.metal_buffer_t(l.WOBuf)
		cSlice[i].wo_type = C.int(l.WOType)

		cSlice[i].ffn_gate = C.metal_buffer_t(l.FFNGateBuf)
		cSlice[i].ffn_gate_type = C.int(l.FFNGateType)
		cSlice[i].ffn_up = C.metal_buffer_t(l.FFNUpBuf)
		cSlice[i].ffn_up_type = C.int(l.FFNUpType)
		cSlice[i].ffn_down = C.metal_buffer_t(l.FFNDownBuf)
		cSlice[i].ffn_down_type = C.int(l.FFNDownType)

		cSlice[i].attn_norm = C.metal_buffer_t(l.AttnNormBuf)
		cSlice[i].ffn_norm = C.metal_buffer_t(l.FFNNormBuf)
	}
	return &PreallocatedLayers{ptr: cPtr, len: len(layers)}
}

// Free releases the C memory allocated for pre-packed layers.
func (p *PreallocatedLayers) Free() {
	if p != nil && p.ptr != nil {
		C.free(p.ptr)
		p.ptr = nil
	}
}

// TransformerParams bundles all parameters and handles for a full-model forward pass.
type TransformerParams struct {
	InitialX, OutLogits                         []float32
	Layers                                      []LayerWeights
	PreallocatedLayers                          *PreallocatedLayers
	OutputNormBuf, OutputWeightBuf              unsafe.Pointer
	OutputWeightType                            int
	NumLayers, Dim, HiddenDim, KVDim, VocabSize int
	NumHeads, NumKVHeads, HeadDim               int
	Pos, Slot, MaxSeq, ActiveContext            int
	NormEps, RopeTheta, AttnScale               float32
}

// ForwardTransformer executes all 40 transformer layers on the GPU in a single CGo call.
func ForwardTransformer(p *TransformerParams) error {
	if !IsAvailable() || p == nil {
		return errUnsupported
	}
	if p.OutputNormBuf == nil || p.OutputWeightBuf == nil {
		return errUnsupported
	}

	var layersPtr *C.metal_layer_weights_t
	if p.PreallocatedLayers != nil && p.PreallocatedLayers.ptr != nil {
		layersPtr = (*C.metal_layer_weights_t)(p.PreallocatedLayers.ptr)
	} else if len(p.Layers) > 0 {
		cLayers := make([]C.metal_layer_weights_t, len(p.Layers))
		for i, l := range p.Layers {
			cLayers[i].wq = C.metal_buffer_t(l.WQBuf)
			cLayers[i].wq_type = C.int(l.WQType)
			cLayers[i].wk = C.metal_buffer_t(l.WKBuf)
			cLayers[i].wk_type = C.int(l.WKType)
			cLayers[i].wv = C.metal_buffer_t(l.WVBuf)
			cLayers[i].wv_type = C.int(l.WVType)
			cLayers[i].wo = C.metal_buffer_t(l.WOBuf)
			cLayers[i].wo_type = C.int(l.WOType)

			cLayers[i].ffn_gate = C.metal_buffer_t(l.FFNGateBuf)
			cLayers[i].ffn_gate_type = C.int(l.FFNGateType)
			cSlice := cLayers
			cSlice[i].ffn_up = C.metal_buffer_t(l.FFNUpBuf)
			cSlice[i].ffn_up_type = C.int(l.FFNUpType)
			cSlice[i].ffn_down = C.metal_buffer_t(l.FFNDownBuf)
			cSlice[i].ffn_down_type = C.int(l.FFNDownType)

			cSlice[i].attn_norm = C.metal_buffer_t(l.AttnNormBuf)
			cSlice[i].ffn_norm = C.metal_buffer_t(l.FFNNormBuf)
		}
		layersPtr = &cLayers[0]
	} else {
		return errUnsupported
	}

	ret := C.metal_forward_transformer(
		(*C.float)(&p.InitialX[0]),
		(*C.float)(&p.OutLogits[0]),
		layersPtr,
		C.metal_buffer_t(p.OutputNormBuf),
		C.metal_buffer_t(p.OutputWeightBuf),
		C.int(p.OutputWeightType),
		C.uint32_t(p.NumLayers),
		C.uint32_t(p.Dim),
		C.uint32_t(p.HiddenDim),
		C.uint32_t(p.KVDim),
		C.uint32_t(p.VocabSize),
		C.uint32_t(p.NumHeads),
		C.uint32_t(p.NumKVHeads),
		C.uint32_t(p.HeadDim),
		C.uint32_t(p.Pos),
		C.uint32_t(p.Slot),
		C.uint32_t(p.MaxSeq),
		C.uint32_t(p.ActiveContext),
		C.float(p.NormEps),
		C.float(p.RopeTheta),
		C.float(p.AttnScale),
	)
	if ret != 0 {
		return fmt.Errorf("metal_forward_transformer failed: %d", int(ret))
	}
	return nil
}
type LayerParams struct {
	X, XNorm, Q, K, V, AttnOut, AttnProj, FFNGate, FFNUp, FFNDown []float32
	AttnNorm, FFNNorm                                             []float32
	WQBuf                                                         unsafe.Pointer
	WQType                                                        int
	WKBuf                                                         unsafe.Pointer
	WKType                                                        int
	WVBuf                                                         unsafe.Pointer
	WVType                                                        int
	WOBuf                                                         unsafe.Pointer
	WOType                                                        int
	FFNGateBuf                                                    unsafe.Pointer
	FFNGateType                                                   int
	FFNUpBuf                                                      unsafe.Pointer
	FFNUpType                                                     int
	FFNDownBuf                                                    unsafe.Pointer
	FFNDownType                                                   int
	LayerIdx                                                      int
	Dim, HiddenDim, KVDim, NumHeads, NumKVHeads, HeadDim          int
	Pos, Slot, MaxSeq, ActiveContext                              int
	NormEps, RopeTheta, AttnScale                                 float32
}

// ForwardLayer dispatches an entire transformer layer in a single CGo call.
func ForwardLayer(p *LayerParams) error {
	if !IsAvailable() || p == nil {
		return errUnsupported
	}
	if p.WQBuf == nil || p.WKBuf == nil || p.WVBuf == nil || p.WOBuf == nil ||
		p.FFNGateBuf == nil || p.FFNUpBuf == nil || p.FFNDownBuf == nil {
		return errUnsupported
	}

	ret := C.metal_forward_layer(
		(*C.float)(&p.X[0]),
		(*C.float)(&p.XNorm[0]),
		(*C.float)(&p.Q[0]),
		(*C.float)(&p.K[0]),
		(*C.float)(&p.V[0]),
		(*C.float)(&p.AttnOut[0]),
		(*C.float)(&p.AttnProj[0]),
		(*C.float)(&p.FFNGate[0]),
		(*C.float)(&p.FFNUp[0]),
		(*C.float)(&p.FFNDown[0]),
		(*C.float)(&p.AttnNorm[0]),
		(*C.float)(&p.FFNNorm[0]),
		C.metal_buffer_t(p.WQBuf), C.int(p.WQType),
		C.metal_buffer_t(p.WKBuf), C.int(p.WKType),
		C.metal_buffer_t(p.WVBuf), C.int(p.WVType),
		C.metal_buffer_t(p.WOBuf), C.int(p.WOType),
		C.metal_buffer_t(p.FFNGateBuf), C.int(p.FFNGateType),
		C.metal_buffer_t(p.FFNUpBuf), C.int(p.FFNUpType),
		C.metal_buffer_t(p.FFNDownBuf), C.int(p.FFNDownType),
		C.uint32_t(p.LayerIdx),
		C.uint32_t(p.Dim),
		C.uint32_t(p.HiddenDim),
		C.uint32_t(p.KVDim),
		C.uint32_t(p.NumHeads),
		C.uint32_t(p.NumKVHeads),
		C.uint32_t(p.HeadDim),
		C.uint32_t(p.Pos),
		C.uint32_t(p.Slot),
		C.uint32_t(p.MaxSeq),
		C.uint32_t(p.ActiveContext),
		C.float(p.NormEps),
		C.float(p.RopeTheta),
		C.float(p.AttnScale),
	)
	if ret != 0 {
		return fmt.Errorf("metal_forward_layer failed: %d", int(ret))
	}
	return nil
}

// MatMulBuf runs GEMV directly using a pre-allocated GPU buffer handle.
func MatMulBuf(quantType int, y, x []float32, wBuf unsafe.Pointer, rows, cols int) error {
	if !IsAvailable() || len(y) < rows || len(x) < cols || wBuf == nil {
		return errUnsupported
	}
	ret := C.metal_gemv_buf(
		C.int(quantType),
		(*C.float)(&y[0]),
		(*C.float)(&x[0]),
		C.metal_buffer_t(wBuf),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemv_buf failed: %d", int(ret))
	}
	return nil
}

// AllocBuffers pre-allocates permanent GPU buffers for intermediate activations and KV-cache.
func AllocBuffers(dim, hiddenDim, kvDim, vocabSize, numLayers, maxSeq int) error {
	ret := C.metal_alloc_buffers(
		C.uint32_t(dim),
		C.uint32_t(hiddenDim),
		C.uint32_t(kvDim),
		C.uint32_t(vocabSize),
		C.uint32_t(numLayers),
		C.uint32_t(maxSeq),
	)
	if ret != 0 {
		return fmt.Errorf("metal_alloc_buffers failed: %d", int(ret))
	}
	return nil
}

// KVWrite writes K and V vectors directly into the GPU-resident KV-cache.
func KVWrite(k, v []float32, layer, slot, maxSeq, kvDim int) error {
	ret := C.metal_kv_write(
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		C.uint32_t(layer),
		C.uint32_t(slot),
		C.uint32_t(maxSeq),
		C.uint32_t(kvDim),
	)
	if ret != 0 {
		return fmt.Errorf("metal_kv_write failed: %d", int(ret))
	}
	return nil
}

// RMSNorm executes vectorized RMS layer normalization on the GPU.
func RMSNorm(out, x, weight []float32, dim int, eps float32) error {
	ret := C.metal_rmsnorm(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&weight[0])),
		C.uint32_t(dim),
		C.float(eps),
	)
	if ret != 0 {
		return fmt.Errorf("metal_rmsnorm failed: %d", int(ret))
	}
	return nil
}

// RoPE applies Rotary Position Embedding to Q and K on the GPU.
func RoPE(q, k []float32, pos, numHeads, numKVHeads, headDim int, theta float32) error {
	ret := C.metal_rope(
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		C.uint32_t(pos),
		C.uint32_t(numHeads),
		C.uint32_t(numKVHeads),
		C.uint32_t(headDim),
		C.float(theta),
	)
	if ret != 0 {
		return fmt.Errorf("metal_rope failed: %d", int(ret))
	}
	return nil
}

// AttentionGQA executes fused multi-head / grouped-query FlashAttention directly on the GPU.
func AttentionGQA(attnOut, q, kCache, vCache []float32, numHeads, numKVHeads, headDim, activeContext int, attnScale float32) error {
	ret := C.metal_attention_gqa(
		(*C.float)(unsafe.Pointer(&attnOut[0])),
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&kCache[0])),
		(*C.float)(unsafe.Pointer(&vCache[0])),
		C.uint32_t(numHeads),
		C.uint32_t(numKVHeads),
		C.uint32_t(headDim),
		C.uint32_t(activeContext),
		C.float(attnScale),
	)
	if ret != 0 {
		return fmt.Errorf("metal_attention_gqa failed: %d", int(ret))
	}
	return nil
}

// SwiGLU computes in-place SiLU(gate) * up on the GPU.
func SwiGLU(gate, up []float32, hiddenDim int) error {
	ret := C.metal_swiglu(
		(*C.float)(unsafe.Pointer(&gate[0])),
		(*C.float)(unsafe.Pointer(&up[0])),
		C.uint32_t(hiddenDim),
	)
	if ret != 0 {
		return fmt.Errorf("metal_swiglu failed: %d", int(ret))
	}
	return nil
}

// AddResidual computes in-place x += proj on the GPU.
func AddResidual(x, proj []float32, dim int) error {
	ret := C.metal_add_residual(
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&proj[0])),
		C.uint32_t(dim),
	)
	if ret != 0 {
		return fmt.Errorf("metal_add_residual failed: %d", int(ret))
	}
	return nil
}

// MatMulF32 executes GPU GEMV for float32 weights.
func MatMulF32(y, x, w []float32, rows, cols int) error {
	if len(y) < rows || len(x) < cols || len(w) < rows*cols {
		return fmt.Errorf("invalid slice dimensions")
	}
	ret := C.metal_gemv_f32(
		(*C.float)(unsafe.Pointer(&y[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemv_f32 failed: %d", int(ret))
	}
	return nil
}

// MatMulF16 executes GPU GEMV for FP16 weights.
func MatMulF16(y, x []float32, rawF16 []byte, rows, cols int) error {
	ret := C.metal_gemv_f16(
		(*C.float)(unsafe.Pointer(&y[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		unsafe.Pointer(&rawF16[0]),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemv_f16 failed: %d", int(ret))
	}
	return nil
}

// MatMulQ4_0 executes GPU GEMV for Q4_0 quantized weights.
func MatMulQ4_0(y, x []float32, rawQ4 []byte, rows, cols int) error {
	ret := C.metal_gemv_q4_0(
		(*C.float)(unsafe.Pointer(&y[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		unsafe.Pointer(&rawQ4[0]),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemv_q4_0 failed: %d", int(ret))
	}
	return nil
}

// MatMulQ8_0 executes GPU GEMV for Q8_0 quantized weights.
func MatMulQ8_0(y, x []float32, rawQ8 []byte, rows, cols int) error {
	ret := C.metal_gemv_q8_0(
		(*C.float)(unsafe.Pointer(&y[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		unsafe.Pointer(&rawQ8[0]),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemv_q8_0 failed: %d", int(ret))
	}
	return nil
}

// MatMulQ4_K executes GPU GEMV for Q4_K quantized weights.
func MatMulQ4_K(y, x []float32, rawQ4K []byte, rows, cols int) error {
	ret := C.metal_gemv_q4_k(
		(*C.float)(unsafe.Pointer(&y[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		unsafe.Pointer(&rawQ4K[0]),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemv_q4_k failed: %d", int(ret))
	}
	return nil
}

// MatMulQ6_K executes GPU GEMV for Q6_K quantized weights.
func MatMulQ6_K(y, x []float32, rawQ6K []byte, rows, cols int) error {
	ret := C.metal_gemv_q6_k(
		(*C.float)(unsafe.Pointer(&y[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		unsafe.Pointer(&rawQ6K[0]),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemv_q6_k failed: %d", int(ret))
	}
	return nil
}

// MatMulBatch executes batched 2D GEMM on the GPU for prompt prefill.
func MatMulBatch(y, x []float32, rawW []byte, qType uint32, batchSize, rows, cols int) error {
	if len(y) < batchSize*rows || len(x) < batchSize*cols {
		return fmt.Errorf("invalid slice dimensions for batch GEMM")
	}

	var ret C.int
	switch qType {
	case 2: // GGMLTypeQ4_0
		ret = C.metal_gemm_q4_0(
			(*C.float)(unsafe.Pointer(&y[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			unsafe.Pointer(&rawW[0]),
			C.uint32_t(batchSize),
			C.uint32_t(rows),
			C.uint32_t(cols),
		)
	case 8: // GGMLTypeQ8_0
		ret = C.metal_gemm_q8_0(
			(*C.float)(unsafe.Pointer(&y[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			unsafe.Pointer(&rawW[0]),
			C.uint32_t(batchSize),
			C.uint32_t(rows),
			C.uint32_t(cols),
		)
	case 12: // GGMLTypeQ4_K
		ret = C.metal_gemm_q4_k(
			(*C.float)(unsafe.Pointer(&y[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			unsafe.Pointer(&rawW[0]),
			C.uint32_t(batchSize),
			C.uint32_t(rows),
			C.uint32_t(cols),
		)
	case 14: // GGMLTypeQ6_K
		ret = C.metal_gemm_q6_k(
			(*C.float)(unsafe.Pointer(&y[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			unsafe.Pointer(&rawW[0]),
			C.uint32_t(batchSize),
			C.uint32_t(rows),
			C.uint32_t(cols),
		)
	default:
		return fmt.Errorf("unsupported quantization type for GPU batch GEMM: %d", qType)
	}

	if ret != 0 {
		return fmt.Errorf("metal_gemm failed with code: %d", int(ret))
	}
	return nil
}
