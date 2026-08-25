//go:build !darwin || !cgo

package metal

import (
	"errors"
	"unsafe"
)

var errUnsupported = errors.New("metal GPU acceleration is only supported on macOS with CGo enabled")

func Init() error {
	return errUnsupported
}

func IsAvailable() bool {
	return false
}

func BeginBatch() {}

func EndBatch() {}

type LayerWeights struct {
	WQBuf, WKBuf, WVBuf, WOBuf          unsafe.Pointer
	WQType, WKType, WVType, WOType      int
	FFNGateBuf, FFNUpBuf, FFNDownBuf    unsafe.Pointer
	FFNGateType, FFNUpType, FFNDownType int
	AttnNormBuf, FFNNormBuf             unsafe.Pointer
}

type PreallocatedLayers struct {
	ptr unsafe.Pointer
	len int
}

func NewPreallocatedLayers(layers []LayerWeights) *PreallocatedLayers {
	return nil
}

func (p *PreallocatedLayers) Free() {}

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

func ForwardTransformer(p *TransformerParams) error {
	return errUnsupported
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

func ForwardLayer(p *LayerParams) error {
	return errUnsupported
}

func CreateBuffer(ptr unsafe.Pointer, bytes int) unsafe.Pointer {
	return nil
}

func ReleaseBuffer(buf unsafe.Pointer) {}

func MatMulBuf(quantType int, y, x []float32, wBuf unsafe.Pointer, rows, cols int) error {
	return errUnsupported
}

func AllocBuffers(dim, hiddenDim, kvDim, vocabSize, numLayers, maxSeq int) error {
	return errUnsupported
}

func KVWrite(k, v []float32, layer, slot, maxSeq, kvDim int) error {
	return errUnsupported
}

func RMSNorm(out, x, weight []float32, dim int, eps float32) error {
	return errUnsupported
}

func RoPE(q, k []float32, pos, numHeads, numKVHeads, headDim int, theta float32) error {
	return errUnsupported
}

func AttentionGQA(attnOut, q, kCache, vCache []float32, numHeads, numKVHeads, headDim, activeContext int, attnScale float32) error {
	return errUnsupported
}

func SwiGLU(gate, up []float32, hiddenDim int) error {
	return errUnsupported
}

func AddResidual(x, proj []float32, dim int) error {
	return errUnsupported
}

func MatMulF32(y, x, w []float32, rows, cols int) error {
	return errUnsupported
}

func MatMulF16(y, x []float32, rawF16 []byte, rows, cols int) error {
	return errUnsupported
}

func MatMulQ4_0(y, x []float32, rawQ4 []byte, rows, cols int) error {
	return errUnsupported
}

func MatMulQ8_0(y, x []float32, rawQ8 []byte, rows, cols int) error {
	return errUnsupported
}

func MatMulQ4_K(y, x []float32, rawQ4K []byte, rows, cols int) error {
	return errUnsupported
}

func MatMulQ6_K(y, x []float32, rawQ6K []byte, rows, cols int) error {
	return errUnsupported
}

func MatMulBatch(y, x []float32, rawW []byte, qType uint32, batchSize, rows, cols int) error {
	return errUnsupported
}
