package distributed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go-inference/pkg/engine"
	"net/http"
	"time"
)

// PipelineStageConfig holds pipeline stage partitioning configuration.
type PipelineStageConfig struct {
	StartLayer int    `json:"start_layer"`
	EndLayer   int    `json:"end_layer"`
	NextURL    string `json:"next_url,omitempty"` // empty if this is the final stage
	IsFinal    bool   `json:"is_final"`
}

// PipelineStage executes a contiguous range of transformer layers and routes to downstream nodes.
type PipelineStage struct {
	Engine     *engine.Engine
	Config     PipelineStageConfig
	HTTPClient *http.Client
}

// NewPipelineStage creates a pipeline parallel stage worker.
func NewPipelineStage(eng *engine.Engine, cfg PipelineStageConfig) *PipelineStage {
	return &PipelineStage{
		Engine:     eng,
		Config:     cfg,
		HTTPClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// ProcessStage executes this node's assigned layer range on the activation vector.
func (s *PipelineStage) ProcessStage(req *PipelineStageRequest, kv *engine.KVCache) (*PipelineStageResponse, error) {
	var inputActivation []float32
	if len(req.Activation) > 0 {
		inputActivation = req.Activation
	} else {
		// First stage: perform embedding lookup
		inputActivation = make([]float32, s.Engine.Config.Dim)
		s.Engine.Weights.ExtractEmbedding(req.TokenID, inputActivation, s.Engine.Config.Dim)
	}

	// Forward through assigned layer partition
	outAct := s.Engine.ForwardLayerRange(inputActivation, s.Config.StartLayer, s.Config.EndLayer, req.Pos, kv)

	if s.Config.IsFinal || s.Config.NextURL == "" {
		// Final stage computes output norm and logits
		logits := s.Engine.ForwardLogits(outAct)
		return &PipelineStageResponse{
			Logits:  logits,
			IsFinal: true,
		}, nil
	}

	// Send intermediate activation to downstream pipeline node
	nextReq := PipelineStageRequest{
		TokenID:    req.TokenID,
		Pos:        req.Pos,
		StartLayer: s.Config.EndLayer + 1,
		Activation: outAct,
		IsFinal:    false,
	}

	data, err := json.Marshal(nextReq)
	if err != nil {
		return nil, fmt.Errorf("marshal pipeline request: %w", err)
	}

	resp, err := s.HTTPClient.Post(s.Config.NextURL+"/v1/dist/pipeline-forward", "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("forward to next pipeline stage (%s): %w", s.Config.NextURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("next pipeline stage returned HTTP %s", resp.Status)
	}

	var res PipelineStageResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("decode pipeline response: %w", err)
	}
	return &res, nil
}
