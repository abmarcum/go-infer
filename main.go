package main

import (
	"bufio"
	"flag"
	_ "embed"
	"fmt"
	"go-inference/pkg/downloader"
	"go-inference/pkg/engine"
	"go-inference/pkg/guardrails"
	"go-inference/pkg/metal"
	"go-inference/pkg/reasoning"
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
		bestOfN        int
		reasoningMode    bool
		enableCalc       bool
		enableGuardrails bool
		cCoreMode        bool
		apiKey           string
		constitutionPath string
	)

	flag.BoolVar(&printVersion, "version", false, "Print version information and exit")
	flag.BoolVar(&printVersion, "v", false, "Print version information and exit (shorthand)")
	flag.StringVar(&modelPath, "model", "", "Path to GGUF model file or Ollama model blob")
	flag.StringVar(&promptText, "prompt", "", "Prompt text to generate completion for")
	flag.StringVar(&serveAddr, "serve", "", "Start HTTP OpenAI & Ollama compatible server on address (e.g. :8080)")
	flag.StringVar(&corsOrigin, "cors-origin", "*", "Allowed CORS origin header for HTTP API")
	flag.StringVar(&apiKey, "api-key", "", "API key required for authenticating requests via Bearer token (or set GO_INFER_API_KEY)")
	flag.StringVar(&constitutionPath, "constitution", "", "Path to custom constitution file for guardrails")
	flag.IntVar(&numThreads, "threads", runtime.NumCPU(), "Number of CPU worker threads for GEMV")
	flag.IntVar(&maxTokens, "max-tokens", 256, "Maximum tokens to generate")
	flag.Float64Var(&temperature, "temp", 0.7, "Sampling temperature (0.0 for greedy)")
	flag.Float64Var(&topP, "top-p", 0.9, "Nucleus Top-P sampling cutoff")
	flag.IntVar(&topK, "top-k", 40, "Top-K sampling cutoff")
	flag.Float64Var(&repPenalty, "rep-penalty", 1.1, "Repetition penalty")
	flag.StringVar(&kvType, "kv-type", "f32", "KV-cache storage precision: f32 (default), q8_0 (2x RAM savings), q4_0 (4x RAM savings)")
	flag.IntVar(&bestOfN, "best-of-n", 1, "Run self-consistency majority voting with N candidate chains")
	flag.BoolVar(&reasoningMode, "reasoning", false, "Enable reasoning mode (optimal hyperparameters for CoT models e.g. DeepSeek-R1)")
	flag.BoolVar(&enableCalc, "calc", false, "Enable embedded math/calculator evaluation tool loop")
	flag.BoolVar(&enableGuardrails, "guardrails", false, "Enable Asimov's Three Laws runtime guardrails on prompt, server, and interactive mode")
	flag.BoolVar(&enableGuardrails, "asimov", false, "Enable Asimov's Three Laws runtime guardrails (alias)")
	flag.BoolVar(&cCoreMode, "c-core", false, "Use hybrid C open-weights inference engine core (inference_core.c)")

	// Distributed inference flags
	flag.StringVar(&distMode, "dist-mode", "none", "Distributed inference mode: none, speculative, pipeline, tensor-parallel")
	flag.StringVar(&draftServer, "draft-server", "", "URL of draft server for speculative decoding (e.g. http://192.168.1.10:8080)")
	flag.IntVar(&draftTokens, "draft-tokens", 4, "Number of speculative draft tokens per verification step")
	flag.StringVar(&pipelineLayers, "pipeline-layers", "", "Layer range for this pipeline stage (e.g. 0-19)")
	flag.StringVar(&pipelineNext, "pipeline-next", "", "URL of downstream pipeline stage server (e.g. http://192.168.1.11:8080)")
	flag.IntVar(&tpRank, "tp-rank", 0, "Tensor parallelism rank of this worker (0, 1, ...)")
	flag.StringVar(&tpPeers, "tp-peers", "", "Comma-separated peer URLs for tensor parallel AllReduce")

	// Check for 'pull' subcommand
	if len(os.Args) >= 2 && os.Args[1] == "pull" {
		if len(os.Args) < 3 {
			fmt.Println("Usage: go-infer pull <huggingface-repo-or-model>")
			fmt.Println("Example: go-infer pull unsloth/Llama-3.2-1B-Instruct-GGUF")
			os.Exit(1)
		}
		target := os.Args[2]
		fmt.Printf(">>> Pulling GGUF model from Hugging Face: %s\n", target)
		savedPath, err := downloader.DownloadHuggingFaceGGUF(target, "models", func(downloaded, total int64, percent, speed float64) {
			fmt.Printf("\rDownloading: %.1f%% (%.2f MB / %.2f MB) @ %.2f MB/s",
				percent, float64(downloaded)/(1024*1024), float64(total)/(1024*1024), speed)
		})
		if err != nil {
			log.Fatalf("\nDownload failed: %v", err)
		}
		fmt.Printf("\nModel successfully saved to: %s\n", savedPath)
		return
	}

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "go-infer - High-Performance GGUF LLM Runtime in Go\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  Pull Model:     go-infer pull <hf-repo/model>\n")
		fmt.Fprintf(os.Stderr, "  Local Prompt:   go-infer [flags] <path-to-gguf> \"<prompt>\"\n")
		fmt.Fprintf(os.Stderr, "  Interactive:    go-infer [flags] <path-to-gguf>\n")
		fmt.Fprintf(os.Stderr, "  HTTP Server:    go-infer --serve :8080 <path-to-gguf>\n")
		fmt.Fprintf(os.Stderr, "  Asimov Guard:   go-infer --guardrails <path-to-gguf> \"<prompt>\"\n")
		fmt.Fprintf(os.Stderr, "  Hybrid C Core:  go-infer --c-core --serve :8080\n")
		fmt.Fprintf(os.Stderr, "  Speculative:    go-infer --dist-mode speculative --draft-server http://draft:8080 <path-to-gguf> \"<prompt>\"\n")
		fmt.Fprintf(os.Stderr, "  Pipeline Stage: go-infer --serve :8080 --pipeline-layers 0-19 --pipeline-next http://stage2:8080 <path-to-gguf>\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	if apiKey == "" {
		apiKey = os.Getenv("GO_INFER_API_KEY")
	}
	if constitutionPath != "" {
		content, err := os.ReadFile(constitutionPath)
		if err != nil {
			log.Fatalf("Failed to read constitution file from %s: %v", constitutionPath, err)
		}
		var laws []string
		for _, line := range strings.Split(string(content), "\n") {
			line = strings.TrimSpace(line)
			line = strings.TrimPrefix(line, "- ")
			line = strings.TrimPrefix(line, "* ")
			if line != "" && !strings.HasPrefix(line, "#") {
				laws = append(laws, line)
			}
		}
		if len(laws) > 0 {
			guardrails.SetCustomConstitution(laws)
			log.Printf("Loaded custom guardrails constitution (%d laws) from: %s", len(laws), constitutionPath)
		}
		enableGuardrails = true
	}

	if printVersion {
		fmt.Printf("go-infer version %s (%s/%s)\n", Version, runtime.GOOS, runtime.GOARCH)
		os.Exit(0)
	}

	// Parse positional arguments if not passed via flags
	args := flag.Args()
	if cCoreMode {
		if len(args) == 1 && promptText == "" && !strings.HasSuffix(args[0], ".gguf") && !strings.Contains(args[0], "/") && !strings.Contains(args[0], "\\") {
			promptText = args[0]
			args = nil
		}
	}
	if modelPath == "" && len(args) > 0 {
		modelPath = args[0]
		args = args[1:]
	}

	if promptText == "" && len(args) > 0 && serveAddr == "" {
		promptText = strings.Join(args, " ")
	}

	// Hybrid C-Core Mode (Asimov Guarded C Inference Engine)
	if cCoreMode {
		if modelPath == "" {
			modelPath = "./models/llama-3-8b-instruct.gguf"
		}
		log.Printf("Initializing Hybrid C-Core inference engine: %s", modelPath)
		cModel, err := InitCModel(modelPath)
		if err != nil {
			log.Fatalf("Failed to initialize C inference engine weights: %v", err)
		}
		defer cModel.Close()

		if serveAddr != "" {
			srv := server.NewServer(nil, filepath.Base(modelPath), serveAddr)
			srv.CORSOrigin = corsOrigin
			srv.APIKey = apiKey
			srv.EnableGuardrails = enableGuardrails
			srv.CoreGenerator = func(prompt string) (string, error) {
				return cModel.Generate(prompt)
			}
			if apiKey != "" {
				fmt.Println("🔒 Bearer token API key authentication enabled")
			}
			if enableGuardrails {
				fmt.Println("🛡️ Asimov Guardrails active on C-Core server")
			}
			fmt.Printf("🚀 Hybrid C+Go Inference Engine running on %s...\n", serveAddr)
			if err := srv.Start(); err != nil {
				log.Fatalf("Server error: %v", err)
			}
			return
		}

		if promptText != "" {
			req := guardrails.GenerationRequest{Prompt: promptText}
			resp := guardrails.ExecutePipeline(req, func(p string) (string, error) {
				return cModel.Generate(p)
			})
			fmt.Printf("\n--- Asimov Guarded Prompt ---\n%s\n\n--- Response ---\n", promptText)
			if resp.Blocked {
				fmt.Printf("[GUARDRAIL BLOCKED: %s]\n", resp.BlockReason)
				if resp.Output != "" {
					fmt.Println(resp.Output)
				}
			} else {
				fmt.Println(resp.Output)
			}
			fmt.Printf("\n[Latency: %d ms | Blocked: %v]\n", resp.LatencyMs, resp.Blocked)
			return
		}

		// Interactive REPL with C-Core
		runCCoreInteractiveREPL(cModel)
		return
	}

	if modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	modelTag := modelPath
	resolvedPath, err := downloader.ResolveModelPath(modelPath)
	if err != nil {
		log.Fatalf("Failed to locate model: %v", err)
	}
	if resolvedPath != modelPath {
		log.Printf("Resolved Ollama model tag '%s' -> %s", modelPath, resolvedPath)
		modelPath = resolvedPath
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
		srv.APIKey = apiKey
		srv.EnableGuardrails = enableGuardrails
		if apiKey != "" {
			fmt.Println("🔒 Bearer token API key authentication enabled")
		}
		if enableGuardrails {
			fmt.Println("🛡️ Asimov Guardrails active on HTTP server (/v1/generate, /v1/chat/completions, /api/generate)")
		}
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
	if reasoningMode {
		params = reasoning.ReasoningParams()
		if maxTokens == 256 {
			maxTokens = 1024
		}
		if temperature > 0 {
			params.Temperature = float32(temperature)
		}
		if topP > 0 {
			params.TopP = float32(topP)
		}
	}

	// Auto-format prompt with chat template if model is Instruct/Chat and prompt doesn't already contain special tags
	formattedPrompt := promptText
	_, hasChatML := eng.Tokenizer.TokenToID["<|im_start|>"]
	_, hasLlamaHeader := eng.Tokenizer.TokenToID["<|start_header_id|>"]
	isChatModel := hasChatML || hasLlamaHeader ||
		strings.Contains(strings.ToLower(modelTag), "instruct") ||
		strings.Contains(strings.ToLower(modelTag), "chat") ||
		strings.Contains(strings.ToLower(modelTag), "qwen") ||
		strings.Contains(strings.ToLower(modelTag), "r1") ||
		strings.Contains(strings.ToLower(modelTag), "deepseek")

	if promptText != "" && !strings.Contains(promptText, "<|") && isChatModel {
		formattedPrompt = eng.FormatChat([]engine.ChatMessage{{Role: "user", Content: promptText}})
	}

	// Self-Consistency Best-of-N Consensus Mode
	if bestOfN > 1 && promptText != "" {
		fmt.Printf("\n--- Self-Consistency Voting (%d paths) ---\nPrompt: %s\n", bestOfN, promptText)
		currentCandidate := -1
		consensus, stats, err := eng.GenerateConsensusWithStream(formattedPrompt, bestOfN, maxTokens, params, func(sampleIdx int, piece string) {
			if sampleIdx != currentCandidate {
				currentCandidate = sampleIdx
				fmt.Printf("\n>>> Candidate %d:\n", sampleIdx+1)
			}
			fmt.Print(piece)
			os.Stdout.Sync()
		}, func(cand *reasoning.CandidateAnswer) {
			fmt.Printf("\n(Extracted: %s)\n", cand.NormAnswer)
		})
		if err != nil {
			log.Fatalf("Consensus error: %v", err)
		}
		fmt.Println("\n═════════════════════════════════════════════════════")
		fmt.Printf("🏆 Consensus Majority Answer: %s\n", consensus.WinningAnswer)
		fmt.Printf("📊 Confidence: %.1f%% (%d/%d votes)\n", consensus.Confidence*100, consensus.Votes, consensus.TotalSamples)
		fmt.Printf("⚡ Throughput: %.2f tok/s across %d tokens\n", stats.TokensPerSecond, stats.GeneratedTokens)
		fmt.Println("═════════════════════════════════════════════════════")
		return
	}

	// Single Prompt Mode
	if promptText != "" {
		if enableGuardrails {
			if err := guardrails.ValidateUserBoundary(guardrails.GenerationRequest{Prompt: promptText}); err != nil {
				fmt.Printf("\n[GUARDRAIL BLOCKED: %v]\n", err)
				return
			}
			formattedPrompt = guardrails.ConstructConstitutionalPrompt(promptText)
		}

		fmt.Printf("\n--- Prompt ---\n%s\n\n--- Response ---\n", promptText)
		var fullResp strings.Builder
		stats, err := eng.GenerateWithTools(formattedPrompt, maxTokens, params, enableCalc, func(token string) bool {
			fmt.Print(token)
			fullResp.WriteString(token)
			return true
		})
		if err != nil {
			log.Fatalf("Generation error: %v", err)
		}
		fmt.Println()

		if enableGuardrails {
			safeOut, blocked, reason := guardrails.CheckOutputGuardrails(fullResp.String())
			if blocked {
				fmt.Printf("\n[GUARDRAIL INTERVENTION: %s]\n%s\n", reason, safeOut)
			}
		}

		fmt.Printf("\n[Prefill: %v | Generation: %v (%d tokens, %.2f tok/s)]\n",
			stats.PrefillDuration, stats.GenerateDuration, stats.GeneratedTokens, stats.TokensPerSecond)
		if reasoningMode {
			ans := reasoning.ExtractAnswer(fullResp.String())
			if ans != "" {
				fmt.Printf("[Extracted Final Answer: %s]\n", ans)
			}
		}
		return
	}

	// Interactive REPL Mode
	runInteractiveREPL(eng, maxTokens, params, enableCalc, enableGuardrails)
}

func runInteractiveREPL(eng *engine.Engine, maxTokens int, params sampler.Params, enableCalc bool, enableGuardrails bool) {
	if enableGuardrails {
		fmt.Println("\n=== Interactive Chat Mode [🛡️ Asimov Guardrails Active] (type 'exit' or Ctrl+C to quit) ===")
	} else {
		fmt.Println("\n=== Interactive Chat Mode (type 'exit' or Ctrl+C to quit) ===")
	}
	scanner := bufio.NewScanner(os.Stdin)
	var messages []engine.ChatMessage

	if enableGuardrails {
		messages = append(messages, engine.ChatMessage{
			Role:    "system",
			Content: guardrails.ConstructConstitutionalPrompt(""),
		})
	}

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

		if enableGuardrails {
			if err := guardrails.ValidateUserBoundary(guardrails.GenerationRequest{Prompt: input}); err != nil {
				fmt.Printf("\n[GUARDRAIL BLOCKED: %v]\n", err)
				continue
			}
		}

		messages = append(messages, engine.ChatMessage{
			Role:    "user",
			Content: input,
		})

		prompt := eng.FormatChat(messages)
		fmt.Print("Assistant > ")

		var assistantResponse strings.Builder
		stats, err := eng.GenerateWithTools(prompt, maxTokens, params, enableCalc, func(token string) bool {
			fmt.Print(token)
			assistantResponse.WriteString(token)
			return true
		})
		if err != nil {
			fmt.Printf("\nError: %v\n", err)
			continue
		}
		fmt.Println()

		replyText := assistantResponse.String()
		if enableGuardrails {
			safeOut, blocked, reason := guardrails.CheckOutputGuardrails(replyText)
			if blocked {
				fmt.Printf("\n[GUARDRAIL INTERVENTION: %s]\n%s\n", reason, safeOut)
				replyText = safeOut
			}
		}

		fmt.Printf("[%.2f tok/s]\n", stats.TokensPerSecond)

		messages = append(messages, engine.ChatMessage{
			Role:    "assistant",
			Content: replyText,
		})
	}
}

func runCCoreInteractiveREPL(cModel *CModelContext) {
	fmt.Println("\n=== Hybrid C+Go Asimov Guarded Chat Mode (type 'exit' or Ctrl+C to quit) ===")
	scanner := bufio.NewScanner(os.Stdin)

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

		req := guardrails.GenerationRequest{Prompt: input}
		resp := guardrails.ExecutePipeline(req, func(prompt string) (string, error) {
			return cModel.Generate(prompt)
		})

		fmt.Print("Assistant > ")
		if resp.Blocked {
			fmt.Printf("[GUARDRAIL BLOCKED: %s]\n", resp.BlockReason)
			if resp.Output != "" {
				fmt.Println(resp.Output)
			}
		} else {
			fmt.Println(resp.Output)
		}
		fmt.Printf("[%d ms latency]\n", resp.LatencyMs)
	}
}
