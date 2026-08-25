package engine

// MemoryArena manages zero-allocation reusable working buffers across forward passes.
type MemoryArena struct {
	X          []float32 // Embedding / hidden state (Dim)
	XB         []float32 // Normalized hidden state (Dim)
	XB2        []float32 // Secondary normalized buffer (Dim)
	Q          []float32 // Query projection (Dim)
	K          []float32 // Key projection (KVDim)
	V          []float32 // Value projection (KVDim)
	AttnOut    []float32 // Attention output before projection (Dim)
	AttnProj   []float32 // Projected attention output (Dim)
	AttnScores []float32 // Softmax scores per head (SeqLen)
	Gate       []float32 // FFN Gate projection (HiddenDim)
	Up         []float32 // FFN Up projection (HiddenDim)
	FFNDown    []float32 // FFN Down projection (Dim)
	Logits     []float32 // Output logits (VocabSize)
}

// NewMemoryArena creates a pre-allocated MemoryArena according to model config.
func NewMemoryArena(cfg ModelConfig) *MemoryArena {
	kvDim := cfg.KVDim()
	seqLen := cfg.SeqLen
	if seqLen <= 0 {
		seqLen = 2048
	}

	return &MemoryArena{
		X:          make([]float32, cfg.Dim),
		XB:         make([]float32, cfg.Dim),
		XB2:        make([]float32, cfg.Dim),
		Q:          make([]float32, cfg.Dim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		AttnOut:    make([]float32, cfg.Dim),
		AttnProj:   make([]float32, cfg.Dim),
		AttnScores: make([]float32, seqLen),
		Gate:       make([]float32, cfg.HiddenDim),
		Up:         make([]float32, cfg.HiddenDim),
		FFNDown:    make([]float32, cfg.Dim),
		Logits:     make([]float32, cfg.VocabSize),
	}
}
