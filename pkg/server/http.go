package server

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"go-inference/pkg/engine"
	"go-inference/pkg/sampler"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

//go:embed web/ui.html
var webUIHTML string

// Server handles OpenAI, Ollama, and Web UI HTTP requests.
type Server struct {
	Engine     *engine.Engine
	ModelName  string
	Port       string
	CORSOrigin string

	// Telemetry & Metrics Counters
	RequestsTotal    uint64
	TokensTotal      uint64
	PrefillMillis    uint64
	GenerationMillis uint64
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

// ResponseFormat defines structured output mode (e.g. json_object).
type ResponseFormat struct {
	Type string `json:"type"`
}

// Tool represents an OpenAI tool specification.
type Tool struct {
	Type     string      `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAI API Types
type OpenAIChatRequest struct {
	Model          string               `json:"model"`
	Messages       []engine.ChatMessage `json:"messages"`
	Stream         bool                 `json:"stream"`
	Temperature    float32              `json:"temperature"`
	TopP           float32              `json:"top_p"`
	MaxTokens      int                  `json:"max_tokens"`
	ResponseFormat *ResponseFormat      `json:"response_format,omitempty"`
	Tools          []Tool               `json:"tools,omitempty"`
	ToolChoice     interface{}          `json:"tool_choice,omitempty"`
}

type OpenAIStreamChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []OpenAIChoiceChunk `json:"choices"`
}

type OpenAIChoiceChunk struct {
	Index        int         `json:"index"`
	Delta        OpenAIDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type OpenAIDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
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
	Message      OpenAIChatMessage  `json:"message"`
	FinishReason string             `json:"finish_reason"`
}

type OpenAIChatMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Embeddings API Types
type OpenAIEmbeddingRequest struct {
	Input interface{} `json:"input"` // string or []string
	Model string      `json:"model"`
}

type OpenAIEmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type OpenAIEmbeddingResponse struct {
	Object string                `json:"object"`
	Data   []OpenAIEmbeddingData `json:"data"`
	Model  string                `json:"model"`
	Usage  OpenAIUsage           `json:"usage"`
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

// Start launches the HTTP API server with secure timeouts and telemetry.
func (s *Server) Start() error {
	mux := http.NewServeMux()

	// Web UI & Metrics
	mux.HandleFunc("/", s.handleWebUI)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/health", s.handleHealth)

	// OpenAI Endpoints
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChatCompletions)
	mux.HandleFunc("/v1/embeddings", s.handleOpenAIEmbeddings)

	// Ollama Endpoints
	mux.HandleFunc("/api/generate", s.handleOllamaGenerate)
	mux.HandleFunc("/api/tags", s.handleOllamaTags)

	// Distributed inference endpoints
	mux.HandleFunc("/v1/dist/pipeline-forward", s.handlePipelineForward)
	mux.HandleFunc("/v1/dist/speculative-draft", s.handleSpeculativeDraft)
	mux.HandleFunc("/v1/dist/tp-reduce", s.handleTPReduce)

	srv := &http.Server{
		Addr:              s.Port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	log.Printf("Server listening on http://0.0.0.0%s (Model: %s)", s.Port, s.ModelName)
	return srv.ListenAndServe()
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

func (s *Server) handleWebUI(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" && r.URL.Path != "/ui" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(webUIHTML))
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	reqs := atomic.LoadUint64(&s.RequestsTotal)
	toks := atomic.LoadUint64(&s.TokensTotal)
	prefillMs := atomic.LoadUint64(&s.PrefillMillis)
	genMs := atomic.LoadUint64(&s.GenerationMillis)

	fmt.Fprintf(w, "# HELP goinfer_requests_total Total number of inference requests\n")
	fmt.Fprintf(w, "# TYPE goinfer_requests_total counter\n")
	fmt.Fprintf(w, "goinfer_requests_total %d\n", reqs)

	fmt.Fprintf(w, "# HELP goinfer_tokens_generated_total Total number of tokens generated\n")
	fmt.Fprintf(w, "# TYPE goinfer_tokens_generated_total counter\n")
	fmt.Fprintf(w, "goinfer_tokens_generated_total %d\n", toks)

	fmt.Fprintf(w, "# HELP goinfer_prefill_duration_seconds Total prompt prefill computation time\n")
	fmt.Fprintf(w, "# TYPE goinfer_prefill_duration_seconds counter\n")
	fmt.Fprintf(w, "goinfer_prefill_duration_seconds %.4f\n", float64(prefillMs)/1000.0)

	fmt.Fprintf(w, "# HELP goinfer_generation_duration_seconds Total token generation computation time\n")
	fmt.Fprintf(w, "# TYPE goinfer_generation_duration_seconds counter\n")
	fmt.Fprintf(w, "goinfer_generation_duration_seconds %.4f\n", float64(genMs)/1000.0)
}

func (s *Server) handleOpenAIEmbeddings(w http.ResponseWriter, r *http.Request) {
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
	var req OpenAIEmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json: %v", err), http.StatusBadRequest)
		return
	}

	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			if str, ok := item.(string); ok {
				inputs = append(inputs, str)
			}
		}
	default:
		http.Error(w, "input must be a string or array of strings", http.StatusBadRequest)
		return
	}

	if len(inputs) == 0 {
		http.Error(w, "input cannot be empty", http.StatusBadRequest)
		return
	}

	totalPromptTokens := 0
	var dataItems []OpenAIEmbeddingData

	for idx, text := range inputs {
		vec, numTokens, err := s.Engine.Embed(text)
		if err != nil {
			http.Error(w, fmt.Sprintf("embedding error: %v", err), http.StatusInternalServerError)
			return
		}
		totalPromptTokens += numTokens
		dataItems = append(dataItems, OpenAIEmbeddingData{
			Object:    "embedding",
			Embedding: vec,
			Index:     idx,
		})
	}

	resp := OpenAIEmbeddingResponse{
		Object: "list",
		Data:   dataItems,
		Model:  s.ModelName,
		Usage: OpenAIUsage{
			PromptTokens: totalPromptTokens,
			TotalTokens:  totalPromptTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
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

	if len(req.Tokens) == 0 {
		http.Error(w, "tokens array must not be empty", http.StatusBadRequest)
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

	atomic.AddUint64(&s.RequestsTotal, 1)
	r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

	var req OpenAIChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json: %v", err), http.StatusBadRequest)
		return
	}

	// Tool definition injection into prompt if tools are specified
	messages := req.Messages
	if len(req.Tools) > 0 {
		var toolDesc strings.Builder
		toolDesc.WriteString("You have access to the following tools:\n")
		for _, t := range req.Tools {
			schemaBytes, _ := json.Marshal(t.Function.Parameters)
			toolDesc.WriteString(fmt.Sprintf("- Function: %s\n  Description: %s\n  Parameters: %s\n", t.Function.Name, t.Function.Description, string(schemaBytes)))
		}
		toolDesc.WriteString("\nIf you choose to invoke a tool, respond with valid JSON: {\"name\": \"<function_name>\", \"arguments\": {<args>}}")
		messages = append([]engine.ChatMessage{{Role: "system", Content: toolDesc.String()}}, messages...)
	}

	prompt := s.Engine.FormatChat(messages)
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

	// Constrained JSON Grammar if requested
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		params.JSONValidator = sampler.NewJSONGrammarValidator()
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

		stats, _ := s.Engine.Generate(prompt, maxTokens, params, func(token string) bool {
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

		if stats != nil {
			atomic.AddUint64(&s.TokensTotal, uint64(stats.GeneratedTokens))
			atomic.AddUint64(&s.PrefillMillis, uint64(stats.PrefillDuration.Milliseconds()))
			atomic.AddUint64(&s.GenerationMillis, uint64(stats.GenerateDuration.Milliseconds()))
		}

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

	if stats != nil {
		atomic.AddUint64(&s.TokensTotal, uint64(stats.GeneratedTokens))
		atomic.AddUint64(&s.PrefillMillis, uint64(stats.PrefillDuration.Milliseconds()))
		atomic.AddUint64(&s.GenerationMillis, uint64(stats.GenerateDuration.Milliseconds()))
	}

	replyText := fullResponse.String()
	var toolCalls []ToolCall

	// Tool call detection in JSON responses
	if len(req.Tools) > 0 && strings.HasPrefix(strings.TrimSpace(replyText), "{") {
		var rawTool struct {
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(replyText)), &rawTool); err == nil && rawTool.Name != "" {
			argsBytes, _ := json.Marshal(rawTool.Arguments)
			toolCalls = append(toolCalls, ToolCall{
				ID:   fmt.Sprintf("call_%d", time.Now().UnixNano()),
				Type: "function",
				Function: ToolCallFunction{
					Name:      rawTool.Name,
					Arguments: string(argsBytes),
				},
			})
		}
	}

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	resp := OpenAIChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.ModelName,
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: OpenAIChatMessage{
					Role:      "assistant",
					Content:   replyText,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
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

	atomic.AddUint64(&s.RequestsTotal, 1)
	r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

	var req OllamaGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json: %v", err), http.StatusBadRequest)
		return
	}

	params := sampler.DefaultParams()
	flusher, _ := w.(http.Flusher)

	w.Header().Set("Content-Type", "application/x-ndjson")

	stats, _ := s.Engine.Generate(req.Prompt, 512, params, func(token string) bool {
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

	if stats != nil {
		atomic.AddUint64(&s.TokensTotal, uint64(stats.GeneratedTokens))
		atomic.AddUint64(&s.PrefillMillis, uint64(stats.PrefillDuration.Milliseconds()))
		atomic.AddUint64(&s.GenerationMillis, uint64(stats.GenerateDuration.Milliseconds()))
	}

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
