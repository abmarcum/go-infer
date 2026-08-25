package distributed

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDistributedSpeculativeVerification(t *testing.T) {
	// Mock Draft Server
	draftServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/dist/speculative-draft" {
			http.NotFound(w, r)
			return
		}
		var req SpeculativeDraftRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		// Returns 3 mock candidate tokens
		res := SpeculativeDraftResponse{
			DraftTokens: []int{101, 102, 103},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(res)
	}))
	defer draftServer.Close()

	coord := NewSpeculativeCoordinator(nil, draftServer.URL, 3)
	tokens, err := coord.QueryDraftServer([]int{1, 2, 3})
	if err != nil {
		t.Fatalf("QueryDraftServer failed: %v", err)
	}

	if len(tokens) != 3 || tokens[0] != 101 || tokens[1] != 102 || tokens[2] != 103 {
		t.Errorf("Draft tokens mismatch: got %v", tokens)
	}
}

func TestDistributedPipelineStageRequestResponse(t *testing.T) {
	// Mock Stage 2 Server
	stage2Server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/dist/pipeline-forward" {
			http.NotFound(w, r)
			return
		}
		var req PipelineStageRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		if req.StartLayer != 20 {
			t.Errorf("Expected StartLayer 20, got %d", req.StartLayer)
		}

		res := PipelineStageResponse{
			Logits:  []float32{0.1, 0.9, 0.0},
			IsFinal: true,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(res)
	}))
	defer stage2Server.Close()

	stage1 := NewPipelineStage(nil, PipelineStageConfig{
		StartLayer: 0,
		EndLayer:   19,
		NextURL:    stage2Server.URL,
		IsFinal:    false,
	})

	if stage1.Config.NextURL != stage2Server.URL {
		t.Errorf("NextURL mismatch: got %s", stage1.Config.NextURL)
	}
}

func TestDistributedTensorParallelAllReduce(t *testing.T) {
	// Mock Peer Rank 1 Server
	peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/dist/tp-reduce" {
			http.NotFound(w, r)
			return
		}
		var req TPReduceRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		// Peer returns partial sum vector [2.0, 2.0, 2.0, 2.0]
		res := TPReduceResponse{
			SumVector: []float32{2.0, 2.0, 2.0, 2.0},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(res)
	}))
	defer peerServer.Close()

	peers := []string{"http://self:8080", peerServer.URL}
	coord := NewTPCoordinator(0, peers)

	localVec := []float32{1.0, 1.0, 1.0, 1.0}
	sumVec, err := coord.AllReduce(localVec)
	if err != nil {
		t.Fatalf("AllReduce failed: %v", err)
	}

	// Expected sum = local (1.0) + peer (2.0) = 3.0
	for i, val := range sumVec {
		if val != 3.0 {
			t.Errorf("AllReduce sum mismatch at %d: got %f, expected 3.0", i, val)
		}
	}
}
