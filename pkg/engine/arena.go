package engine

// MemoryArena manages zero-allocation reusable working buffers across forward passes.
type MemoryArena struct {
	X          []float32 // Embedding / hidden state (Dim)
	XB         []float32 // Normalized hidden state (Dim)
	XB2        []float32 // Secondary normalized buffer (Dim)
	Q          []float32 // Query projection (Dim or AttnDim/QDim)
	K          []float32 // Key projection (KVDim)
	V          []float32 // Value projection (KVDim)
	QRaw       []float32 // Raw Q+Gate projection for hybrid models (e.g. 12288)
	AttnGate   []float32 // Post-attention gate (AttnDim, e.g. 6144)
	AttnOut    []float32 // Attention output before projection (Dim or AttnDim)
	AttnProj   []float32 // Projected attention output (Dim)
	AttnScores []float32 // Softmax scores per head (SeqLen)
	Gate       []float32 // FFN Gate projection (HiddenDim)
	Up         []float32 // FFN Up projection (HiddenDim)
	FFNDown    []float32 // FFN Down projection (Dim)
	Logits     []float32 // Output logits (VocabSize)

	// SSM working buffers for hybrid architectures
	SSMQKV     []float32 // SSM QKV projection (SSMChannels, e.g. 10240)
	SSMGate    []float32 // SSM Gate projection (SSMInnerSize, e.g. 6144)
	SSMOut     []float32 // SSM Output state (SSMInnerSize, e.g. 6144)
	SSMConvOut []float32 // SSM 1D convolution output (SSMChannels, e.g. 10240)
	SSMAlpha   []float32 // Timestep dt projection (SSMTimeStepRank, e.g. 48)
	SSMBeta    []float32 // DeltaNet beta projection (SSMTimeStepRank, e.g. 48)
}

// NewMemoryArena creates a pre-allocated MemoryArena according to model config.
func NewMemoryArena(cfg ModelConfig) *MemoryArena {
	kvDim := cfg.KVDim()
	seqLen := cfg.SeqLen
	if seqLen <= 0 {
		seqLen = 2048
	}

	qDim := cfg.Dim
	if cfg.AttnDim() > qDim {
		qDim = cfg.AttnDim()
	}
	if cfg.Architecture == "qwen35" && 12288 > qDim {
		qDim = 12288
	}
	rawQDim := qDim

	attnOutDim := cfg.Dim
	if cfg.AttnDim() > attnOutDim {
		attnOutDim = cfg.AttnDim()
	}

	ssmInner := cfg.SSMInnerSize
	if ssmInner <= 0 {
		ssmInner = 6144
	}
	ssmState := cfg.SSMStateSize
	if ssmState <= 0 {
		ssmState = 128
	}
	ssmGroups := cfg.SSMGroupCount
	if ssmGroups <= 0 {
		ssmGroups = 16
	}
	ssmChannels := ssmInner + 2*ssmGroups*ssmState
	ssmRank := cfg.SSMTimeStepRank
	if ssmRank <= 0 {
		ssmRank = 48
	}

	return &MemoryArena{
		X:          make([]float32, cfg.Dim),
		XB:         make([]float32, cfg.Dim),
		XB2:        make([]float32, cfg.Dim),
		Q:          make([]float32, qDim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		QRaw:       make([]float32, rawQDim),
		AttnGate:   make([]float32, attnOutDim),
		AttnOut:    make([]float32, attnOutDim),
		AttnProj:   make([]float32, cfg.Dim),
		AttnScores: make([]float32, seqLen),
		Gate:       make([]float32, cfg.HiddenDim),
		Up:         make([]float32, cfg.HiddenDim),
		FFNDown:    make([]float32, cfg.Dim),
		Logits:     make([]float32, cfg.VocabSize),

		SSMQKV:     make([]float32, ssmChannels),
		SSMGate:    make([]float32, ssmInner),
		SSMOut:     make([]float32, ssmInner),
		SSMConvOut: make([]float32, ssmChannels),
		SSMAlpha:   make([]float32, ssmRank),
		SSMBeta:    make([]float32, ssmRank),
	}
}
