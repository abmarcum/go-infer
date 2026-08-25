package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go-inference/pkg/engine"
	"go-inference/pkg/sampler"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type OllamaRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	Options map[string]interface{} `json:"options"`
}

type OllamaResponse struct {
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
}

func main() {
	var (
		modelBlob  string
		modelName  string
		prompt     string
		maxTokens  int
		threads    int
		ollamaURL  string
	)

	flag.StringVar(&modelBlob, "blob", "", "Path to Ollama GGUF blob file")
	flag.StringVar(&modelName, "ollama-model", "qwen3.6:35b-a3b", "Ollama model name (e.g. qwen3.6:35b-a3b)")
	flag.StringVar(&prompt, "prompt", "Explain goroutines in Go in two concise sentences.", "Benchmark prompt")
	flag.IntVar(&maxTokens, "max-tokens", 30, "Max tokens to generate")
	flag.IntVar(&threads, "threads", runtime.NumCPU(), "CPU threads for Go inference")
	flag.StringVar(&ollamaURL, "ollama-url", "http://localhost:11434/api/generate", "Ollama API endpoint")

	flag.Parse()

	// If blob not provided, attempt to locate blob from ~/.ollama/models/blobs
	if modelBlob == "" && len(flag.Args()) > 0 {
		modelBlob = flag.Args()[0]
	}
	if modelBlob == "" {
		home, _ := os.UserHomeDir()
		// Try default blob
		candidate := filepath.Join(home, ".ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d")
		if _, err := os.Stat(candidate); err == nil {
			modelBlob = candidate
		}
	}

	fmt.Println("===============================================================")
	fmt.Println("       INFERENCE BENCHMARK: Pure Go vs. Native Ollama          ")
	fmt.Println("===============================================================")
	fmt.Printf("Prompt:     %q\n", prompt)
	fmt.Printf("Max Tokens: %d\n", maxTokens)
	fmt.Printf("CPU Cores:  %d\n\n", threads)

	// 1. Benchmark Native Ollama
	fmt.Println(">>> [1/2] Running Native Ollama (llama.cpp / GPU / C++)...")
	ollamaText, ollamaTPS, ollamaDur, err := benchmarkOllama(ollamaURL, modelName, prompt, maxTokens)
	if err != nil {
		fmt.Printf("Ollama test skipped / failed: %v\n", err)
	} else {
		fmt.Printf("Ollama Output:\n%s\n", strings.TrimSpace(ollamaText))
		fmt.Printf("Ollama Speed:  %.2f tokens/sec (Time: %v)\n\n", ollamaTPS, ollamaDur)
	}

	// 2. Benchmark Pure Go Engine
	if modelBlob != "" {
		fmt.Println(">>> [2/2] Running GoInfer Engine (Pure Go + Metal)...")
		goText, goTPS, goPrefill, goGen, err := benchmarkGoEngine(modelBlob, prompt, maxTokens, threads)
		if err != nil {
			log.Fatalf("GoInfer failed: %v", err)
		}
		fmt.Printf("Go Engine Output:\n%s\n", strings.TrimSpace(goText))
		fmt.Printf("Go Speed:      %.2f tokens/sec (Prefill: %v, Gen: %v)\n\n", goTPS, goPrefill, goGen)

		// 3. Comparison Table
		fmt.Println("===============================================================")
		fmt.Println("                      BENCHMARK SUMMARY                        ")
		fmt.Println("===============================================================")
		fmt.Printf("%-25s | %-15s | %-15s\n", "Engine", "Speed (tokens/s)", "Total Latency")
		fmt.Println("---------------------------------------------------------------")
		if err == nil {
			fmt.Printf("%-25s | %-15.2f | %-15v\n", "Ollama Native (C++/GPU)", ollamaTPS, ollamaDur)
		}
		fmt.Printf("%-25s | %-15.2f | %-15v\n", "go-infer (Pure Go)", goTPS, goPrefill+goGen)
		fmt.Println("===============================================================")
	}
}

func benchmarkOllama(url, model, prompt string, maxTokens int) (string, float64, time.Duration, error) {
	reqBody := OllamaRequest{
		Model:  model,
		Prompt: prompt,
		Stream: true,
		Options: map[string]interface{}{
			"num_predict": maxTokens,
			"temperature": 0.0,
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, 0, err
	}

	start := time.Now()
	resp, err := http.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		return "", 0, 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", 0, 0, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var sb strings.Builder
	reader := bufio.NewReader(resp.Body)
	var lastChunk OllamaResponse

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}
		var chunk OllamaResponse
		if err := json.Unmarshal(line, &chunk); err == nil {
			sb.WriteString(chunk.Response)
			if chunk.Done {
				lastChunk = chunk
				break
			}
		}
	}
	dur := time.Since(start)

	tps := 0.0
	if lastChunk.EvalDuration > 0 {
		tps = float64(lastChunk.EvalCount) / (float64(lastChunk.EvalDuration) / 1e9)
	} else if dur.Seconds() > 0 {
		tps = float64(maxTokens) / dur.Seconds()
	}

	return sb.String(), tps, dur, nil
}

func benchmarkGoEngine(modelBlob, prompt string, maxTokens, threads int) (string, float64, time.Duration, time.Duration, error) {
	eng, err := engine.LoadModel(modelBlob, threads)
	if err != nil {
		return "", 0, 0, 0, err
	}
	defer eng.Close()

	params := sampler.Params{Temperature: 0.0}
	var sb strings.Builder

	stats, err := eng.Generate(prompt, maxTokens, params, func(token string) bool {
		sb.WriteString(token)
		return true
	})
	if err != nil {
		return "", 0, 0, 0, err
	}

	return sb.String(), stats.TokensPerSecond, stats.PrefillDuration, stats.GenerateDuration, nil
}
