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
	"os"
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
	if os.Getenv("DISABLE_METAL") == "1" {
		return false
	}
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
	BQBuf, BKBuf, BVBuf                 unsafe.Pointer
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

		cSlice[i].bq = C.metal_buffer_t(l.BQBuf)
		cSlice[i].bk = C.metal_buffer_t(l.BKBuf)
		cSlice[i].bv = C.metal_buffer_t(l.BVBuf)
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
	OutToken                                    *uint32
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

			cSlice[i].bq = C.metal_buffer_t(l.BQBuf)
			cSlice[i].bk = C.metal_buffer_t(l.BKBuf)
			cSlice[i].bv = C.metal_buffer_t(l.BVBuf)
		}
		layersPtr = &cLayers[0]
	} else {
		return errUnsupported
	}

	var logitsPtr *C.float
	if len(p.OutLogits) > 0 {
		logitsPtr = (*C.float)(&p.OutLogits[0])
	}
	var tokenPtr *C.uint32_t
	if p.OutToken != nil {
		tokenPtr = (*C.uint32_t)(unsafe.Pointer(p.OutToken))
	}

	ret := C.metal_forward_transformer(
		(*C.float)(&p.InitialX[0]),
		logitsPtr,
		tokenPtr,
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

// Qwen35LayerWeights holds persistent GPU buffer handles and quantization types for a hybrid layer.
type Qwen35LayerWeights struct {
	IsSSM bool

	AttnNormBuf unsafe.Pointer
	FFNNormBuf  unsafe.Pointer

	// SSM layer weights
	SSMGateBuf   unsafe.Pointer
	SSMGateType  int
	SSMQKVBuf    unsafe.Pointer
	SSMQKVType   int
	SSMConv1DBuf unsafe.Pointer
	SSMAlphaBuf  unsafe.Pointer
	SSMAlphaType int
	SSMBetaBuf   unsafe.Pointer
	SSMBetaType  int
	SSMDTBiasBuf unsafe.Pointer
	SSMABuf      unsafe.Pointer
	SSMNormBuf   unsafe.Pointer
	SSMOutBuf    unsafe.Pointer
	SSMOutType   int

	// Attention layer weights
	WQBuf    unsafe.Pointer
	WQType   int
	WKBuf    unsafe.Pointer
	WKType   int
	WVBuf    unsafe.Pointer
	WVType   int
	WOBuf    unsafe.Pointer
	WOType   int
	QNormBuf unsafe.Pointer
	KNormBuf unsafe.Pointer

	// FFN weights (shared by both)
	FFNGateBuf  unsafe.Pointer
	FFNGateType int
	FFNUpBuf    unsafe.Pointer
	FFNUpType   int
	FFNDownBuf  unsafe.Pointer
	FFNDownType int
}

// PreallocatedQwen35Layers stores a C-pinned array of Qwen3.5 layer weight handles.
type PreallocatedQwen35Layers struct {
	ptr unsafe.Pointer
	len int
}

// NewPreallocatedQwen35Layers pre-packs Qwen35LayerWeights into a persistent C-allocated buffer.
func NewPreallocatedQwen35Layers(layers []Qwen35LayerWeights) *PreallocatedQwen35Layers {
	if len(layers) == 0 {
		return nil
	}
	size := len(layers) * int(unsafe.Sizeof(C.metal_qwen35_layer_weights_t{}))
	cPtr := C.malloc(C.size_t(size))
	cSlice := (*[1 << 20]C.metal_qwen35_layer_weights_t)(cPtr)[:len(layers):len(layers)]
	for i, l := range layers {
		cSlice[i].is_ssm = C.bool(l.IsSSM)
		cSlice[i].attn_norm = C.metal_buffer_t(l.AttnNormBuf)
		cSlice[i].ffn_norm = C.metal_buffer_t(l.FFNNormBuf)

		cSlice[i].ssm_gate = C.metal_buffer_t(l.SSMGateBuf)
		cSlice[i].ssm_gate_type = C.int(l.SSMGateType)
		cSlice[i].ssm_qkv = C.metal_buffer_t(l.SSMQKVBuf)
		cSlice[i].ssm_qkv_type = C.int(l.SSMQKVType)
		cSlice[i].ssm_conv1d = C.metal_buffer_t(l.SSMConv1DBuf)
		cSlice[i].ssm_alpha = C.metal_buffer_t(l.SSMAlphaBuf)
		cSlice[i].ssm_alpha_type = C.int(l.SSMAlphaType)
		cSlice[i].ssm_beta = C.metal_buffer_t(l.SSMBetaBuf)
		cSlice[i].ssm_beta_type = C.int(l.SSMBetaType)
		cSlice[i].ssm_dt_bias = C.metal_buffer_t(l.SSMDTBiasBuf)
		cSlice[i].ssm_a = C.metal_buffer_t(l.SSMABuf)
		cSlice[i].ssm_norm = C.metal_buffer_t(l.SSMNormBuf)
		cSlice[i].ssm_out = C.metal_buffer_t(l.SSMOutBuf)
		cSlice[i].ssm_out_type = C.int(l.SSMOutType)

		cSlice[i].wq = C.metal_buffer_t(l.WQBuf)
		cSlice[i].wq_type = C.int(l.WQType)
		cSlice[i].wk = C.metal_buffer_t(l.WKBuf)
		cSlice[i].wk_type = C.int(l.WKType)
		cSlice[i].wv = C.metal_buffer_t(l.WVBuf)
		cSlice[i].wv_type = C.int(l.WVType)
		cSlice[i].wo = C.metal_buffer_t(l.WOBuf)
		cSlice[i].wo_type = C.int(l.WOType)
		cSlice[i].q_norm = C.metal_buffer_t(l.QNormBuf)
		cSlice[i].k_norm = C.metal_buffer_t(l.KNormBuf)

		cSlice[i].ffn_gate = C.metal_buffer_t(l.FFNGateBuf)
		cSlice[i].ffn_gate_type = C.int(l.FFNGateType)
		cSlice[i].ffn_up = C.metal_buffer_t(l.FFNUpBuf)
		cSlice[i].ffn_up_type = C.int(l.FFNUpType)
		cSlice[i].ffn_down = C.metal_buffer_t(l.FFNDownBuf)
		cSlice[i].ffn_down_type = C.int(l.FFNDownType)
	}
	return &PreallocatedQwen35Layers{ptr: cPtr, len: len(layers)}
}

// Free releases the C memory allocated for pre-packed Qwen 3.5 layers.
func (p *PreallocatedQwen35Layers) Free() {
	if p != nil && p.ptr != nil {
		C.free(p.ptr)
		p.ptr = nil
	}
}

// Qwen35TransformerParams bundles all parameters and handles for a full Qwen 3.5 forward pass.
type Qwen35TransformerParams struct {
	InitialX, OutLogits                         []float32
	TokenEmbdBuf                                unsafe.Pointer
	TokenID                                     int
	OutToken                                    *uint32
	Layers                                      []Qwen35LayerWeights
	PreallocatedLayers                          *PreallocatedQwen35Layers
	OutputNormBuf, OutputWeightBuf              unsafe.Pointer
	OutputWeightType                            int
	NumLayers, Dim, HiddenDim                   int
	SSMInner, SSMChannels, SSMStateSize         int
	SSMGroups, SSMRank                          int
	KVDim, VocabSize                            int
	NumHeads, NumKVHeads, HeadDim, RopeDim      int
	Pos, Slot, MaxSeq, ActiveContext            int
	NormEps, RopeTheta, AttnScale               float32
}

// ForwardQwen35 executes all 65 hybrid transformer layers on the GPU in a single CGo call.
func ForwardQwen35(p *Qwen35TransformerParams) error {
	if !IsAvailable() || p == nil {
		return errUnsupported
	}
	if p.OutputNormBuf == nil || p.OutputWeightBuf == nil {
		return errUnsupported
	}

	var layersPtr *C.metal_qwen35_layer_weights_t
	if p.PreallocatedLayers != nil && p.PreallocatedLayers.ptr != nil {
		layersPtr = (*C.metal_qwen35_layer_weights_t)(p.PreallocatedLayers.ptr)
	} else {
		return errUnsupported
	}

	var initialXPtr *C.float
	if len(p.InitialX) > 0 {
		initialXPtr = (*C.float)(&p.InitialX[0])
	}
	var tokenEmbdBufPtr C.metal_buffer_t
	if p.TokenEmbdBuf != nil {
		tokenEmbdBufPtr = C.metal_buffer_t(p.TokenEmbdBuf)
	}

	var logitsPtr *C.float
	if len(p.OutLogits) > 0 {
		logitsPtr = (*C.float)(&p.OutLogits[0])
	}
	var tokenPtr *C.uint32_t
	if p.OutToken != nil {
		tokenPtr = (*C.uint32_t)(unsafe.Pointer(p.OutToken))
	}

	ret := C.metal_forward_qwen35(
		initialXPtr,
		tokenEmbdBufPtr,
		C.int32_t(p.TokenID),
		logitsPtr,
		tokenPtr,
		layersPtr,
		C.metal_buffer_t(p.OutputNormBuf),
		C.metal_buffer_t(p.OutputWeightBuf),
		C.int(p.OutputWeightType),
		C.uint32_t(p.NumLayers),
		C.uint32_t(p.Dim),
		C.uint32_t(p.HiddenDim),
		C.uint32_t(p.SSMInner),
		C.uint32_t(p.SSMChannels),
		C.uint32_t(p.SSMStateSize),
		C.uint32_t(p.SSMGroups),
		C.uint32_t(p.SSMRank),
		C.uint32_t(p.KVDim),
		C.uint32_t(p.VocabSize),
		C.uint32_t(p.NumHeads),
		C.uint32_t(p.NumKVHeads),
		C.uint32_t(p.HeadDim),
		C.uint32_t(p.RopeDim),
		C.uint32_t(p.Pos),
		C.uint32_t(p.Slot),
		C.uint32_t(p.MaxSeq),
		C.uint32_t(p.ActiveContext),
		C.float(p.NormEps),
		C.float(p.RopeTheta),
		C.float(p.AttnScale),
	)
	if ret != 0 {
		return fmt.Errorf("metal_forward_qwen35 failed: %d", int(ret))
	}
	return nil
}

// AllocQwen35Buffers pre-allocates permanent GPU buffers for Qwen 3.5 hybrid architecture.
func AllocQwen35Buffers(dim, hiddenDim, ssmInner, ssmChannels, ssmRank, ssmStateSize, kvDim, vocabSize, numLayers, maxSeq int) error {
	ret := C.metal_alloc_qwen35_buffers(
		C.uint32_t(dim),
		C.uint32_t(hiddenDim),
		C.uint32_t(ssmInner),
		C.uint32_t(ssmChannels),
		C.uint32_t(ssmRank),
		C.uint32_t(ssmStateSize),
		C.uint32_t(kvDim),
		C.uint32_t(vocabSize),
		C.uint32_t(numLayers),
		C.uint32_t(maxSeq),
	)
	if ret != 0 {
		return fmt.Errorf("metal_alloc_qwen35_buffers failed: %d", int(ret))
	}
	return nil
}

// ResetSSMState zeroes out recurrent SSM states in GPU memory.
func ResetSSMState() {
	if IsAvailable() {
		C.metal_reset_ssm_state()
	}
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

// MatMulBatchBuf runs batched GEMM directly using a pre-allocated GPU buffer handle.
func MatMulBatchBuf(quantType int, y, x []float32, wBuf unsafe.Pointer, batchSize, rows, cols int) error {
	if !IsAvailable() || len(y) < batchSize*rows || len(x) < batchSize*cols || wBuf == nil {
		return errUnsupported
	}
	ret := C.metal_gemm_buf(
		C.int(quantType),
		(*C.float)(&y[0]),
		(*C.float)(&x[0]),
		C.metal_buffer_t(wBuf),
		C.uint32_t(batchSize),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemm_buf failed: %d", int(ret))
	}
	return nil
}

// MatMulFusedGateUpBatchBuf runs batched fused Gate+Up+SwiGLU directly using pre-allocated GPU buffer handles.
func MatMulFusedGateUpBatchBuf(quantType int, y, x []float32, gateBuf, upBuf unsafe.Pointer, batchSize, rows, cols int) error {
	if !IsAvailable() || len(y) < batchSize*rows || len(x) < batchSize*cols || gateBuf == nil || upBuf == nil {
		return errUnsupported
	}
	ret := C.metal_gemm_fused_gate_up_buf(
		C.int(quantType),
		(*C.float)(&y[0]),
		(*C.float)(&x[0]),
		C.metal_buffer_t(gateBuf),
		C.metal_buffer_t(upBuf),
		C.uint32_t(batchSize),
		C.uint32_t(rows),
		C.uint32_t(cols),
	)
	if ret != 0 {
		return fmt.Errorf("metal_gemm_fused_gate_up_buf failed: %d", int(ret))
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

// RMSNormBatch computes RMSNorm across a batch of vectors on the GPU.
func RMSNormBatch(out, x, weight []float32, dim int, eps float32, batchSize int) error {
	if !IsAvailable() || len(out) < batchSize*dim || len(x) < batchSize*dim || len(weight) < dim || batchSize <= 0 {
		return errUnsupported
	}
	ret := C.metal_rmsnorm_batch(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&weight[0])),
		C.uint32_t(dim),
		C.float(eps),
		C.uint32_t(batchSize),
	)
	if ret != 0 {
		return fmt.Errorf("metal_rmsnorm_batch failed: %d", int(ret))
	}
	return nil
}

// ResidualRMSNormBatch fuses residual addition (x += proj) and subsequent RMSNorm (out_norm = rmsnorm(x)) across a batch.
func ResidualRMSNormBatch(x, proj, outNorm, weight []float32, dim int, eps float32, batchSize int) error {
	if !IsAvailable() || len(x) < batchSize*dim || len(proj) < batchSize*dim || len(outNorm) < batchSize*dim || len(weight) < dim || batchSize <= 0 {
		return errUnsupported
	}
	ret := C.metal_residual_rmsnorm_batch(
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&proj[0])),
		(*C.float)(unsafe.Pointer(&outNorm[0])),
		(*C.float)(unsafe.Pointer(&weight[0])),
		C.uint32_t(dim),
		C.float(eps),
		C.uint32_t(batchSize),
	)
	if ret != 0 {
		return fmt.Errorf("metal_residual_rmsnorm_batch failed: %d", int(ret))
	}
	return nil
}

// EmbedLookupQ4KBatch extracts embeddings for a batch of token IDs directly on the GPU.
func EmbedLookupQ4KBatch(outX []float32, wEmbd unsafe.Pointer, tokenIDs []int, dim int) error {
	batchSize := len(tokenIDs)
	if !IsAvailable() || len(outX) < batchSize*dim || wEmbd == nil || batchSize <= 0 {
		return errUnsupported
	}
	uTokens := make([]uint32, batchSize)
	for i, t := range tokenIDs {
		uTokens[i] = uint32(t)
	}
	ret := C.metal_embed_lookup_q4_k_batch(
		(*C.float)(unsafe.Pointer(&outX[0])),
		C.metal_buffer_t(wEmbd),
		(*C.uint32_t)(unsafe.Pointer(&uTokens[0])),
		C.uint32_t(dim),
		C.uint32_t(batchSize),
	)
	if ret != 0 {
		return fmt.Errorf("metal_embed_lookup_q4_k_batch failed: %d", int(ret))
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

// Conv1DBatch executes 1D causal convolution across a batch on the GPU.
func Conv1DBatch(out, in, state, convWeight []float32, kernelSize, channels, batchSize int) error {
	var statePtr *C.float
	if len(state) > 0 {
		statePtr = (*C.float)(unsafe.Pointer(&state[0]))
	}
	ret := C.metal_conv1d_batch(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])),
		statePtr,
		(*C.float)(unsafe.Pointer(&convWeight[0])),
		C.uint32_t(kernelSize),
		C.uint32_t(channels),
		C.uint32_t(batchSize),
	)
	if ret != 0 {
		return fmt.Errorf("metal_conv1d_batch failed: %d", int(ret))
	}
	return nil
}

// SSMRecurrenceBatch executes Gated DeltaNet recurrent SSM updates across a batch on the GPU.
func SSMRecurrenceBatch(ssmOut, convOut, alpha, beta, dtBias, ssmA, normW, gate, ssmState []float32, ssmInner, ssmStateSize, ssmGroups, ssmRank int, eps float32, batchSize, ssmChannels int) error {
	var betaPtr, dtBiasPtr, ssmAPtr, normWPtr *C.float
	if len(beta) > 0 {
		betaPtr = (*C.float)(unsafe.Pointer(&beta[0]))
	}
	if len(dtBias) > 0 {
		dtBiasPtr = (*C.float)(unsafe.Pointer(&dtBias[0]))
	}
	if len(ssmA) > 0 {
		ssmAPtr = (*C.float)(unsafe.Pointer(&ssmA[0]))
	}
	if len(normW) > 0 {
		normWPtr = (*C.float)(unsafe.Pointer(&normW[0]))
	}
	ret := C.metal_ssm_recurrence_batch(
		(*C.float)(unsafe.Pointer(&ssmOut[0])),
		(*C.float)(unsafe.Pointer(&convOut[0])),
		(*C.float)(unsafe.Pointer(&alpha[0])),
		betaPtr,
		dtBiasPtr,
		ssmAPtr,
		normWPtr,
		(*C.float)(unsafe.Pointer(&gate[0])),
		(*C.float)(unsafe.Pointer(&ssmState[0])),
		C.uint32_t(ssmInner),
		C.uint32_t(ssmStateSize),
		C.uint32_t(ssmGroups),
		C.uint32_t(ssmRank),
		C.float(eps),
		C.uint32_t(batchSize),
		C.uint32_t(ssmChannels),
	)
	if ret != 0 {
		return fmt.Errorf("metal_ssm_recurrence_batch failed: %d", int(ret))
	}
	return nil
}

// AttentionGQABatch executes online FlashAttention with GQA for a batch on the GPU.
func AttentionGQABatch(attnOut, q, kCache, vCache []float32, numHeads, numKVHeads, headDim, startPos, maxSeq int, attnScale float32, batchSize int) error {
	ret := C.metal_attention_gqa_batch(
		(*C.float)(unsafe.Pointer(&attnOut[0])),
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&kCache[0])),
		(*C.float)(unsafe.Pointer(&vCache[0])),
		C.uint32_t(numHeads),
		C.uint32_t(numKVHeads),
		C.uint32_t(headDim),
		C.uint32_t(startPos),
		C.uint32_t(maxSeq),
		C.float(attnScale),
		C.uint32_t(batchSize),
	)
	if ret != 0 {
		return fmt.Errorf("metal_attention_gqa_batch failed: %d", int(ret))
	}
	return nil
}

// KVWriteBatch writes batched key and value vectors into the persistent KV cache on the GPU.
func KVWriteBatch(kCache, vCache, k, v []float32, startPos, maxSeq, kvDim, batchSize int) error {
	ret := C.metal_kv_write_batch(
		(*C.float)(unsafe.Pointer(&kCache[0])),
		(*C.float)(unsafe.Pointer(&vCache[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		C.uint32_t(startPos),
		C.uint32_t(maxSeq),
		C.uint32_t(kvDim),
		C.uint32_t(batchSize),
	)
	if ret != 0 {
		return fmt.Errorf("metal_kv_write_batch failed: %d", int(ret))
	}
	return nil
}

// SplitQGateBatch deinterleaves Q and Gate for a batch on the GPU.
func SplitQGateBatch(qOut, gateOut, qGateIn []float32, numTokens int) error {
	ret := C.metal_qwen35_split_q_gate_batch(
		(*C.float)(unsafe.Pointer(&qOut[0])),
		(*C.float)(unsafe.Pointer(&gateOut[0])),
		(*C.float)(unsafe.Pointer(&qGateIn[0])),
		C.uint32_t(numTokens),
	)
	if ret != 0 {
		return fmt.Errorf("metal_qwen35_split_q_gate_batch failed: %d", int(ret))
	}
	return nil
}

// RoPENormBatch performs per-head RMSNorm and rotary position embedding for a batch on the GPU.
func RoPENormBatch(q, k, qNormW, kNormW []float32, startPos, numHeads, numKVHeads, headDim, ropeDim int, theta, eps float32, batchSize int) error {
	ret := C.metal_rope_norm_batch(
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&qNormW[0])),
		(*C.float)(unsafe.Pointer(&kNormW[0])),
		C.uint32_t(startPos),
		C.uint32_t(numHeads),
		C.uint32_t(numKVHeads),
		C.uint32_t(headDim),
		C.uint32_t(ropeDim),
		C.float(theta),
		C.float(eps),
		C.uint32_t(batchSize),
	)
	if ret != 0 {
		return fmt.Errorf("metal_rope_norm_batch failed: %d", int(ret))
	}
	return nil
}

// AttnGate applies post-attention element-wise sigmoid gating on the GPU.
func AttnGate(attnOut, gate []float32, totalElements int) error {
	ret := C.metal_qwen35_attn_gate(
		(*C.float)(unsafe.Pointer(&attnOut[0])),
		(*C.float)(unsafe.Pointer(&gate[0])),
		C.uint32_t(totalElements),
	)
	if ret != 0 {
		return fmt.Errorf("metal_qwen35_attn_gate failed: %d", int(ret))
	}
	return nil
}

