package main

import (
	"bufio"
	"flag"
	_ "embed"
	"fmt"
	"go-inference/pkg/engine"
	"go-inference/pkg/metal"
	"go-inference/pkg/sampler"
	"go-inference/pkg/server"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

//go:embed VERSION
var embeddedVersion string

// Version is the application version, defaulted from the root VERSION file and overrideable via ldflags
var Version = strings.TrimSpace(embeddedVersion)

func main() {
	var (
		printVersion   bool
		modelPath      string
		promptText     string
		serveAddr      string
		numThreads     int
		maxTokens      int
		temperature    float64
		topP           float64
		topK           int
		repPenalty     float64
		kvType         string
		corsOrigin     string
		distMode       string
		draftServer    string
		draftTokens    int
		pipelineLayers string
		pipelineNext   string
		tpRank         int
		tpPeers        string
	)

	flag.BoolVar(&printVersion, "version", false, "Print version information and exit")
	flag.BoolVar(&printVersion, "v", false, "Print version information and exit (shorthand)")
	flag.StringVar(&modelPath, "model", "", "Path to GGUF model file or Ollama model blob")
	flag.StringVar(&promptText, "prompt", "", "Prompt text to generate completion for")
	flag.StringVar(&serveAddr, "serve", "", "Start HTTP OpenAI & Ollama compatible server on address (e.g. :8080)")
	flag.StringVar(&corsOrigin, "cors-origin", "*", "Allowed CORS origin header for HTTP API")
	flag.IntVar(&numThreads, "threads", runtime.NumCPU(), "Number of CPU worker threads for GEMV")
	flag.IntVar(&maxTokens, "max-tokens", 256, "Maximum tokens to generate")
	flag.Float64Var(&temperature, "temp", 0.7, "Sampling temperature (0.0 for greedy)")
	flag.Float64Var(&topP, "top-p", 0.9, "Nucleus Top-P sampling cutoff")
	flag.IntVar(&topK, "top-k", 40, "Top-K sampling cutoff")
	flag.Float64Var(&repPenalty, "rep-penalty", 1.1, "Repetition penalty")
	flag.StringVar(&kvType, "kv-type", "f32", "KV-cache storage precision: f32 (default), q8_0 (2x RAM savings), q4_0 (4x RAM savings)")

	// Distributed inference flags
	flag.StringVar(&distMode, "dist-mode", "none", "Distributed inference mode: none, speculative, pipeline, tensor-parallel")
	flag.StringVar(&draftServer, "draft-server", "", "URL of draft server for speculative decoding (e.g. http://192.168.1.10:8080)")
	flag.IntVar(&draftTokens, "draft-tokens", 4, "Number of speculative draft tokens per verification step")
	flag.StringVar(&pipelineLayers, "pipeline-layers", "", "Layer range for this pipeline stage (e.g. 0-19)")
	flag.StringVar(&pipelineNext, "pipeline-next", "", "URL of downstream pipeline stage server (e.g. http://192.168.1.11:8080)")
	flag.IntVar(&tpRank, "tp-rank", 0, "Tensor parallelism rank of this worker (0, 1, ...)")
	flag.StringVar(&tpPeers, "tp-peers", "", "Comma-separated peer URLs for tensor parallel AllReduce")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "go-infer - High-Performance GGUF LLM Runtime in Go\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  Local Prompt:   go-infer [flags] <path-to-gguf> \"<prompt>\"\n")
		fmt.Fprintf(os.Stderr, "  Interactive:    go-infer [flags] <path-to-gguf>\n")
		fmt.Fprintf(os.Stderr, "  HTTP Server:    go-infer --serve :8080 <path-to-gguf>\n")
		fmt.Fprintf(os.Stderr, "  Speculative:    go-infer --dist-mode speculative --draft-server http://draft:8080 <path-to-gguf> \"<prompt>\"\n")
		fmt.Fprintf(os.Stderr, "  Pipeline Stage: go-infer --serve :8080 --pipeline-layers 0-19 --pipeline-next http://stage2:8080 <path-to-gguf>\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	if printVersion {
		fmt.Printf("go-infer version %s (%s/%s)\n", Version, runtime.GOOS, runtime.GOARCH)
		os.Exit(0)
	}

	// Parse positional arguments if not passed via flags
	args := flag.Args()
	if modelPath == "" && len(args) > 0 {
		modelPath = args[0]
		args = args[1:]
	}

	if promptText == "" && len(args) > 0 && serveAddr == "" {
		promptText = strings.Join(args, " ")
	}

	if modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	log.Printf("Loading GGUF model from: %s", modelPath)
	eng, err := engine.LoadModel(modelPath, numThreads)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer eng.Close()

	log.Printf("Model initialized successfully:")
	log.Printf("  • Architecture: (%d layers, dim=%d, hidden_dim=%d)", eng.Config.NumLayers, eng.Config.Dim, eng.Config.HiddenDim)
	log.Printf("  • Attention:    %d heads (KV heads=%d, head_dim=%d)", eng.Config.NumHeads, eng.Config.NumKVHeads, eng.Config.HeadDim())
	log.Printf("  • Vocabulary:   %d tokens (BOS=%d, EOS=%d, EOT=%d)", eng.Config.VocabSize, eng.Config.BosID, eng.Config.EosID, eng.Config.EotID)
	log.Printf("  • Context:      %d tokens max context", eng.Config.SeqLen)
	log.Printf("  • Threads:      %d CPU workers", numThreads)
	if metal.IsAvailable() {
		log.Printf("  • GPU Backend:  Apple Metal (Accelerated)")
	} else {
		log.Printf("  • GPU Backend:  Disabled (CPU Software)")
	}

	// Server Mode
	if serveAddr != "" {
		modelName := filepath.Base(modelPath)
		srv := server.NewServer(eng, modelName, serveAddr)
		srv.CORSOrigin = corsOrigin
		if err := srv.Start(); err != nil {
			log.Fatalf("Server error: %v", err)
		}
		return
	}

	params := sampler.Params{
		Temperature: float32(temperature),
		TopP:        float32(topP),
		TopK:        topK,
		RepPenalty:  float32(repPenalty),
	}

	// Single Prompt Mode
	if promptText != "" {
		fmt.Printf("\n--- Prompt ---\n%s\n\n--- Response ---\n", promptText)
		stats, err := eng.Generate(promptText, maxTokens, params, func(token string) bool {
			fmt.Print(token)
			return true
		})
		if err != nil {
			log.Fatalf("Generation error: %v", err)
		}
		fmt.Println()
		fmt.Printf("\n[Prefill: %v | Generation: %v (%d tokens, %.2f tok/s)]\n",
			stats.PrefillDuration, stats.GenerateDuration, stats.GeneratedTokens, stats.TokensPerSecond)
		return
	}

	// Interactive REPL Mode
	runInteractiveREPL(eng, maxTokens, params)
}

func runInteractiveREPL(eng *engine.Engine, maxTokens int, params sampler.Params) {
	fmt.Println("\n=== Interactive Chat Mode (type 'exit' or Ctrl+C to quit) ===")
	scanner := bufio.NewScanner(os.Stdin)
	var messages []engine.ChatMessage

	for {
		fmt.Print("\nUser > ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "exit" || input == "quit" {
			break
		}

		messages = append(messages, engine.ChatMessage{
			Role:    "user",
			Content: input,
		})

		prompt := eng.FormatChat(messages)
		fmt.Print("Assistant > ")

		var assistantResponse strings.Builder
		stats, err := eng.Generate(prompt, maxTokens, params, func(token string) bool {
			fmt.Print(token)
			assistantResponse.WriteString(token)
			return true
		})
		if err != nil {
			fmt.Printf("\nError: %v\n", err)
			continue
		}
		fmt.Println()
		fmt.Printf("[%.2f tok/s]\n", stats.TokensPerSecond)

		messages = append(messages, engine.ChatMessage{
			Role:    "assistant",
			Content: assistantResponse.String(),
		})
	}
}
