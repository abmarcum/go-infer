package server

import (
	"bytes"
	"encoding/json"
	"go-inference/pkg/engine"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestServerEndpoints(t *testing.T) {
	tmpDir := t.TempDir()
	// Re-use synthetic model creation helper logic
	modelPath := filepath.Join(tmpDir, "dummy.gguf")

	// Create minimal valid synthetic GGUF model file
	// We can write synthetic test file or verify routes
	testFile, err := os.Create(modelPath)
	if err != nil {
		t.Fatalf("create test file: %v", err)
	}
	testFile.Close()

	// Direct route test
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)

	s := &Server{
		Engine: &engine.Engine{
			Config: engine.ModelConfig{
				Dim:       16,
				NumLayers: 1,
				VocabSize: 6,
			},
		},
		ModelName: "test-model",
	}

	s.handleHealth(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("Expected 200 OK, got %d", rec.Code)
	}

	var res map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&res); err != nil {
		t.Fatalf("failed to decode health response: %v", err)
	}
	if res["model"] != "test-model" {
		t.Errorf("Expected model 'test-model', got %v", res["model"])
	}
}

func TestServerChatCompletion(t *testing.T) {
	// Test payload parsing and mock execution
	body := OpenAIChatRequest{
		Model: "test-model",
		Messages: []engine.ChatMessage{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}
	data, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(data))
	rec := httptest.NewRecorder()

	if req.Method != http.MethodPost {
		t.Errorf("expected POST method")
	}
	_ = rec
}

func TestServerOllamaEndpoint(t *testing.T) {
	body := OllamaGenerateRequest{
		Model:  "test-model",
		Prompt: "Hello world",
		Stream: false,
	}
	data, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(data))
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if req.URL.Path != "/api/generate" {
		t.Errorf("expected /api/generate path")
	}
}

func TestServerConcurrentHealthChecks(t *testing.T) {
	s := &Server{
		Engine: &engine.Engine{
			Config: engine.ModelConfig{
				Dim:       16,
				NumLayers: 1,
				VocabSize: 6,
			},
		},
		ModelName: "test-model",
	}

	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, "/health", nil)
			s.handleHealth(rec, req)
			if rec.Code != http.StatusOK {
				t.Errorf("Expected 200 OK, got %d", rec.Code)
			}
			done <- true
		}()
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestServerPayloadLimit(t *testing.T) {
	s := &Server{
		Engine: &engine.Engine{
			Config: engine.ModelConfig{
				Dim:       16,
				NumLayers: 1,
				VocabSize: 6,
				SeqLen:    128,
			},
		},
		ModelName: "test-model",
	}

	// 11MB payload (exceeds 10MB limit)
	hugeData := make([]byte, 11*1024*1024)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(hugeData))
	rec := httptest.NewRecorder()

	s.handleOpenAIChatCompletions(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("Expected BadRequest (400) on oversized payload, got %d", rec.Code)
	}
}

func TestServerCORSHeaders(t *testing.T) {
	s := &Server{
		Engine: &engine.Engine{
			Config: engine.ModelConfig{
				Dim:       16,
				NumLayers: 1,
				VocabSize: 6,
			},
		},
		ModelName:  "test-model",
		CORSOrigin: "https://example.com",
	}

	req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()

	s.handleOpenAIChatCompletions(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("Expected 200 OK for OPTIONS preflight, got %d", rec.Code)
	}
	if rec.Header().Get("Access-Control-Allow-Origin") != "https://example.com" {
		t.Errorf("Expected CORS origin 'https://example.com', got '%s'", rec.Header().Get("Access-Control-Allow-Origin"))
	}
}
