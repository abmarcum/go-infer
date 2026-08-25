package distributed

// DistMode defines the multi-server coordination strategy.
type DistMode string

const (
	DistModeNone           DistMode = "none"
	DistModeSpeculative    DistMode = "speculative"
	DistModePipeline       DistMode = "pipeline"
	DistModeTensorParallel DistMode = "tensor-parallel"
)

// PipelineStageRequest transfers intermediate hidden activations between pipeline nodes.
type PipelineStageRequest struct {
	TokenID    int       `json:"token_id"`
	Pos        int       `json:"pos"`
	StartLayer int       `json:"start_layer"`
	EndLayer   int       `json:"end_layer"`
	Activation []float32 `json:"activation"`
	IsFinal    bool      `json:"is_final"`
}

// PipelineStageResponse returns the processed activation or final logits.
type PipelineStageResponse struct {
	Activation []float32 `json:"activation,omitempty"`
	Logits     []float32 `json:"logits,omitempty"`
	IsFinal    bool      `json:"is_final"`
}

// SpeculativeDraftRequest requests draft tokens from a lightweight draft server.
type SpeculativeDraftRequest struct {
	Tokens    []int `json:"tokens"`
	NumTokens int   `json:"num_tokens"`
}

// SpeculativeDraftResponse returns the generated speculative candidate tokens.
type SpeculativeDraftResponse struct {
	DraftTokens []int `json:"draft_tokens"`
}

// TPReduceRequest sends a partial matrix dot product vector for AllReduce summation.
type TPReduceRequest struct {
	Rank   int       `json:"rank"`
	StepID uint64    `json:"step_id"`
	Vector []float32 `json:"vector"`
}

// TPReduceResponse returns the globally summed vector across all peer ranks.
type TPReduceResponse struct {
	SumVector []float32 `json:"sum_vector"`
}
