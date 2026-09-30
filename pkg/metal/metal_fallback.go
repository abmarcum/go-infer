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
	BQBuf, BKBuf, BVBuf                 unsafe.Pointer
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

func ForwardTransformer(p *TransformerParams) error {
	return errUnsupported
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

func NewPreallocatedQwen35Layers(layers []Qwen35LayerWeights) *PreallocatedQwen35Layers {
	return nil
}

func (p *PreallocatedQwen35Layers) Free() {}

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

func ForwardQwen35(p *Qwen35TransformerParams) error {
	return errUnsupported
}

func AllocQwen35Buffers(dim, hiddenDim, ssmInner, ssmChannels, ssmRank, ssmStateSize, kvDim, vocabSize, numLayers, maxSeq int) error {
	return errUnsupported
}

func ResetSSMState() {}

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

func MatMulBatchBuf(quantType int, y, x []float32, wBuf unsafe.Pointer, batchSize, rows, cols int) error {
	return errUnsupported
}

func MatMulFusedGateUpBatchBuf(quantType int, y, x []float32, gateBuf, upBuf unsafe.Pointer, batchSize, rows, cols int) error {
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

func RMSNormBatch(out, x, weight []float32, dim int, eps float32, batchSize int) error {
	return errUnsupported
}

func ResidualRMSNormBatch(x, proj, outNorm, weight []float32, dim int, eps float32, batchSize int) error {
	return errUnsupported
}

func EmbedLookupQ4KBatch(outX []float32, wEmbd unsafe.Pointer, tokenIDs []int, dim int) error {
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

func Conv1DBatch(out, in, state, convWeight []float32, kernelSize, channels, batchSize int) error {
	return errUnsupported
}

func SSMRecurrenceBatch(ssmOut, convOut, alpha, beta, dtBias, ssmA, normW, gate, ssmState []float32, ssmInner, ssmStateSize, ssmGroups, ssmRank int, eps float32, batchSize, ssmChannels int) error {
	return errUnsupported
}

func AttentionGQABatch(attnOut, q, kCache, vCache []float32, numHeads, numKVHeads, headDim, startPos, maxSeq int, attnScale float32, batchSize int) error {
	return errUnsupported
}

func KVWriteBatch(kCache, vCache, k, v []float32, startPos, maxSeq, kvDim, batchSize int) error {
	return errUnsupported
}

func SplitQGateBatch(qOut, gateOut, qGateIn []float32, numTokens int) error {
	return errUnsupported
}

func RoPENormBatch(q, k, qNormW, kNormW []float32, startPos, numHeads, numKVHeads, headDim, ropeDim int, theta, eps float32, batchSize int) error {
	return errUnsupported
}

func AttnGate(attnOut, gate []float32, totalElements int) error {
	return errUnsupported
}

