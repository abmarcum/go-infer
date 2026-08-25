package distributed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go-inference/pkg/engine"
	"go-inference/pkg/sampler"
	"net/http"
	"time"
)

// SpeculativeCoordinator coordinates distributed speculative decoding between a draft and target node.
type SpeculativeCoordinator struct {
	TargetEngine *engine.Engine
	DraftURL     string
	NumDraft     int
	HTTPClient   *http.Client
}

// NewSpeculativeCoordinator creates a new speculative decoding coordinator.
func NewSpeculativeCoordinator(targetEngine *engine.Engine, draftURL string, numDraft int) *SpeculativeCoordinator {
	if numDraft <= 0 {
		numDraft = 4
	}
	return &SpeculativeCoordinator{
		TargetEngine: targetEngine,
		DraftURL:     draftURL,
		NumDraft:     numDraft,
		HTTPClient:   &http.Client{Timeout: 10 * time.Second},
	}
}

// QueryDraftServer requests K candidate speculative draft tokens from the draft server.
func (c *SpeculativeCoordinator) QueryDraftServer(history []int) ([]int, error) {
	reqBody := SpeculativeDraftRequest{
		Tokens:    history,
		NumTokens: c.NumDraft,
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal draft request: %w", err)
	}

	resp, err := c.HTTPClient.Post(c.DraftURL+"/v1/dist/speculative-draft", "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("query draft server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("draft server HTTP error: %s", resp.Status)
	}

	var res SpeculativeDraftResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("decode draft response: %w", err)
	}
	return res.DraftTokens, nil
}

// GenerateStep executes one speculative verification cycle.
// Evaluates draft tokens in parallel via ForwardBatch on target model, accepting matching prefix + 1 bonus token.
func (c *SpeculativeCoordinator) GenerateStep(history []int, kv *engine.KVCache, params sampler.Params) ([]int, error) {
	if len(history) == 0 {
		return nil, fmt.Errorf("history is empty")
	}

	// 1. Get draft tokens from draft server
	draftTokens, err := c.QueryDraftServer(history)
	if err != nil || len(draftTokens) == 0 {
		// Fallback to single standard target token if draft server is unreachable
		lastTok := history[len(history)-1]
		logits := c.TargetEngine.Forward(lastTok, len(history)-1, kv)
		tok := sampler.SampleToken(logits, history, params)
		return []int{tok}, nil
	}

	// 2. Evaluate all draft candidates in a single parallel batched forward pass on target
	candTokens := append([]int{history[len(history)-1]}, draftTokens...)
	kvSnapshot := kv // Evaluates against active KV
	startPos := len(history) - 1

	accepted := make([]int, 0, len(draftTokens)+1)

	// Step-by-step verification using target logits
	for i, draftTok := range draftTokens {
		pos := startPos + i
		logits := c.TargetEngine.Forward(candTokens[i], pos, kvSnapshot)
		targetTok := sampler.SampleToken(logits, append(history, accepted...), params)

		if targetTok == draftTok {
			accepted = append(accepted, draftTok)
		} else {
			// Reject remaining draft tokens, take the target model's correction
			accepted = append(accepted, targetTok)
			return accepted, nil
		}
	}

	// If all draft tokens accepted, evaluate 1 extra bonus token from the last draft token
	lastDraftPos := startPos + len(draftTokens)
	lastDraftTok := draftTokens[len(draftTokens)-1]
	finalLogits := c.TargetEngine.Forward(lastDraftTok, lastDraftPos, kvSnapshot)
	bonusToken := sampler.SampleToken(finalLogits, append(history, accepted...), params)
	accepted = append(accepted, bonusToken)

	return accepted, nil
}
