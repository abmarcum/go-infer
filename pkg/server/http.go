package server

import (
	"encoding/json"
	"fmt"
	"go-inference/pkg/engine"
	"go-inference/pkg/sampler"
	"log"
	"net/http"
	"strings"
	"time"
)

// Server handles OpenAI and Ollama compatible HTTP requests.
type Server struct {
	Engine     *engine.Engine
	ModelName  string
	Port       string
	CORSOrigin string
}

// NewServer creates a new HTTP server instance.
func NewServer(eng *engine.Engine, modelName, port string) *Server {
	if !strings.HasPrefix(port, ":") {
		port = ":" + port
	}
	return &Server{
		Engine:     eng,
		ModelName:  modelName,
		Port:       port,
		CORSOrigin: "*",
	}
}

// OpenAI API Types
type OpenAIChatRequest struct {
	Model       string               `json:"model"`
	Messages    []engine.ChatMessage `json:"messages"`
	Stream      bool                 `json:"stream"`
	Temperature float32              `json:"temperature"`
	TopP        float32              `json:"top_p"`
	MaxTokens   int                  `json:"max_tokens"`
}

type OpenAIStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []OpenAIChoiceChunk `json:"choices"`
}

type OpenAIChoiceChunk struct {
	Index        int              `json:"index"`
	Delta        OpenAIDelta      `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
}

type OpenAIDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type OpenAIChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage    `json:"usage"`
}

type OpenAIChoice struct {
	Index        int                `json:"index"`
	Message      engine.ChatMessage `json:"message"`
	FinishReason string             `json:"finish_reason"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Ollama API Types
type OllamaGenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaGenerateChunk struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"`
	Done      bool   `json:"done"`
}

// Start launches the HTTP API server.
func (s *Server) Start() error {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChatCompletions)
	mux.HandleFunc("/api/generate", s.handleOllamaGenerate)
	mux.HandleFunc("/api/tags", s.handleOllamaTags)

	// Distributed inference endpoints
	mux.HandleFunc("/v1/dist/pipeline-forward", s.handlePipelineForward)
	mux.HandleFunc("/v1/dist/speculative-draft", s.handleSpeculativeDraft)
	mux.HandleFunc("/v1/dist/tp-reduce", s.handleTPReduce)

	log.Printf("Server listening on http://0.0.0.0%s (Model: %s)", s.Port, s.ModelName)
	return http.ListenAndServe(s.Port, mux)
}

func (s *Server) setCORS(w http.ResponseWriter) {
	origin := s.CORSOrigin
	if origin == "" {
		origin = "*"
	}
	w.Header().Set("Access-Control-Allow-Origin", origin)
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
}

func (s *Server) handlePipelineForward(w http.ResponseWriter, r *http.Request) {
	s.setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

	var req struct {
		TokenID    int       `json:"token_id"`
		Pos        int       `json:"pos"`
		StartLayer int       `json:"start_layer"`
		EndLayer   int       `json:"end_layer"`
		Activation []float32 `json:"activation"`
		IsFinal    bool      `json:"is_final"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	kv := s.Engine.NewKVCache()
	outAct := s.Engine.ForwardLayerRange(req.Activation, req.StartLayer, req.EndLayer, req.Pos, kv)

	w.Header().Set("Content-Type", "application/json")
	if req.IsFinal {
		logits := s.Engine.ForwardLogits(outAct)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"logits":   logits,
			"is_final": true,
		})
	} else {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"activation": outAct,
			"is_final":   false,
		})
	}
}

func (s *Server) handleSpeculativeDraft(w http.ResponseWriter, r *http.Request) {
	s.setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

	var req struct {
		Tokens    []int `json:"tokens"`
		NumTokens int   `json:"num_tokens"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.NumTokens <= 0 {
		req.NumTokens = 4
	}
	if req.NumTokens > 16 {
		req.NumTokens = 16
	}

	kv := s.Engine.NewKVCache()
	history := append([]int{}, req.Tokens...)
	draftTokens := make([]int, 0, req.NumTokens)

	pos := len(history) - 1
	curTok := history[pos]
	params := sampler.Params{Temperature: 0.7, TopK: 40, TopP: 0.9}

	for i := 0; i < req.NumTokens; i++ {
		logits := s.Engine.Forward(curTok, pos, kv)
		nextTok := sampler.SampleToken(logits, history, params)
		draftTokens = append(draftTokens, nextTok)
		history = append(history, nextTok)
		curTok = nextTok
		pos++
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"draft_tokens": draftTokens,
	})
}

func (s *Server) handleTPReduce(w http.ResponseWriter, r *http.Request) {
	s.setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

	var req struct {
		Rank   int       `json:"rank"`
		StepID uint64    `json:"step_id"`
		Vector []float32 `json:"vector"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Echo back partial vector for peer reduction
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"sum_vector": req.Vector,
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.setCORS(w)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"model":  s.ModelName,
		"layers": s.Engine.Config.NumLayers,
		"dim":    s.Engine.Config.Dim,
		"vocab":  s.Engine.Config.VocabSize,
	})
}

func (s *Server) handleOllamaTags(w http.ResponseWriter, r *http.Request) {
	s.setCORS(w)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"models": []map[string]interface{}{
			{
				"name":        s.ModelName,
				"modified_at": time.Now().Format(time.RFC3339),
				"size":        len(s.Engine.Reader.Data),
			},
		},
	})
}

func (s *Server) handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

	var req OpenAIChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json: %v", err), http.StatusBadRequest)
		return
	}

	prompt := s.Engine.FormatChat(req.Messages)
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	if maxTokens > s.Engine.Config.SeqLen {
		maxTokens = s.Engine.Config.SeqLen
	}

	params := sampler.Params{
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        40,
		RepPenalty:  1.1,
	}
	if params.Temperature <= 0 {
		params.Temperature = 0.7
	}
	if params.TopP <= 0 {
		params.TopP = 0.9
	}

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}

		reqID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

		_, _ = s.Engine.Generate(prompt, maxTokens, params, func(token string) bool {
			chunk := OpenAIStreamChunk{
				ID:      reqID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   s.ModelName,
				Choices: []OpenAIChoiceChunk{
					{
						Index: 0,
						Delta: OpenAIDelta{Content: token},
					},
				},
			}
			b, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
			return true
		})

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	// Non-streaming response
	var fullResponse strings.Builder
	stats, err := s.Engine.Generate(prompt, maxTokens, params, func(token string) bool {
		fullResponse.WriteString(token)
		return true
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	resp := OpenAIChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.ModelName,
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: engine.ChatMessage{
					Role:    "assistant",
					Content: fullResponse.String(),
				},
				FinishReason: "stop",
			},
		},
		Usage: OpenAIUsage{
			PromptTokens:     stats.PromptTokens,
			CompletionTokens: stats.GeneratedTokens,
			TotalTokens:      stats.PromptTokens + stats.GeneratedTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	s.setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

	var req OllamaGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json: %v", err), http.StatusBadRequest)
		return
	}

	params := sampler.DefaultParams()
	flusher, _ := w.(http.Flusher)

	w.Header().Set("Content-Type", "application/x-ndjson")

	_, _ = s.Engine.Generate(req.Prompt, 512, params, func(token string) bool {
		chunk := OllamaGenerateChunk{
			Model:     s.ModelName,
			CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
			Response:  token,
			Done:      false,
		}
		b, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "%s\n", b)
		if flusher != nil {
			flusher.Flush()
		}
		return true
	})

	finalChunk := OllamaGenerateChunk{
		Model:     s.ModelName,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Done:      true,
	}
	b, _ := json.Marshal(finalChunk)
	fmt.Fprintf(w, "%s\n", b)
	if flusher != nil {
		flusher.Flush()
	}
}
