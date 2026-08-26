<div align="center">

<img src="assets/logo.jpg" alt="go-infer logo" width="300" />

# go-infer

**High-Performance, Pure Go LLM Inference Runtime with Apple Metal GPU & Distributed Acceleration**

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows%20|%20iOS-blue)](README.md)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Stdlib)-brightgreen)](go.mod)

</div>

---

## 📌 Executive Summary

**`go-infer`** is a lightweight, zero-external-dependency LLM inference engine written in pure Go with native Apple Metal GPU acceleration for macOS and an optimized parallel CPU runtime for Linux and Windows.

It parses GGUF binary files directly, executes quantized matrix math (`Q2_K` through `Q8_0`), serves OpenAI- and Ollama-compatible streaming APIs, provides an embedded real-time Web UI dashboard, and scales horizontally across multiple machines with Distributed Speculative Decoding and Pipeline Parallelism.

---

## ⚡ Key Features at a Glance

| Category | Highlights |
| :--- | :--- |
| **🚀 Compute & GPU Acceleration** | • **Apple Metal GPU Pipeline**: 8-way SIMD, fused Gate-Up SwiGLU, 100% VRAM residency<br>• **Pure Go CPU Fallback**: Multithreaded worker pool on Linux, Windows & iOS<br>• **Quantized Matrix Math**: Direct vector dot-products (`Q2_K`, `Q3_K`, `Q4_0`, `Q4_K`, `Q6_K`, `Q8_0`) |
| **🧠 Memory & Context Optimization** | • **Prefix & Prompt Cache (Radix)**: Instant KV reuse dropping prefill latency to **~0 ms**<br>• **Quantized KV-Cache**: 8-bit & 4-bit attention storage cutting context RAM by **4×**<br>• **Paged KV Memory**: Block-table allocation eliminating memory fragmentation |
| **🛠️ Native APIs & Developer Tooling** | • **Hugging Face Downloader (`pull`)**: Stream models directly with auto-discovery<br>• **Embedded Dark-Mode Web UI**: Built-in streaming chat at `http://localhost:8080`<br>• **OpenAI & Ollama APIs**: SSE & NDJSON streaming endpoints with tool calling<br>• **JSON Grammar Mode**: Guaranteed valid JSON via logit masking<br>• **Embeddings API**: Dense vector extraction for RAG & vector databases<br>• **Prometheus Telemetry**: Real-time throughput & latency metrics (`/metrics`) |
| **🌐 Multi-Server Distributed Scaling** | • **Distributed Speculative Decoding**: 2×–3× speedup over standard LAN/Wi-Fi<br>• **Pipeline Parallelism**: Split 70B–405B models across multiple nodes (1 hop/tok)<br>• **Tensor Parallelism**: Split matrix computation with AllReduce synchronization |
| **📦 Deployment & Packaging** | • **Zero External Go Dependencies**: Standard library only (no CGo on Linux/Windows)<br>• **Docker & Compose**: Hardened, unprivileged container images with health checks<br>• **Linux Distribution Packages**: Native Debian (`.deb`) and Red Hat (`.rpm`) installers |

---

## 🚀 Quick Start (30 Seconds)

```bash
# 1. Build the binary
make build

# 2. Pull a model directly from Hugging Face
./goinfer pull unsloth/Llama-3.2-1B-Instruct-GGUF

# 3. Launch HTTP server & Web Chat UI
./goinfer --serve :8080 models/llama-3.2-1b-instruct.Q4_K_M.gguf

# 4. Open http://localhost:8080 in your browser or chat via CLI
./goinfer models/llama-3.2-1b-instruct.Q4_K_M.gguf "Explain goroutines in Go in two sentences."
```

---

## Directory Structure

```
go-infer/
├── main.go                       # Main CLI & Server entrypoint
├── Dockerfile                    # Multi-stage container definition
├── docker-compose.yml            # Container orchestration config
├── Makefile                      # Build, test, packaging, and benchmark targets
├── go.mod                        # Go module definition
├── assets/
│   └── logo.jpg                  # Project logo asset
├── cmd/
│   ├── bench/                    # Side-by-side benchmark runner (Pure Go vs Native Ollama)
│   ├── package/                  # Universal pure-Go Debian (.deb) & RPM package generator
│   └── quantize/                 # Standalone model quantizer utility (F32/F16 -> Q4_0/Q8_0)
├── packaging/                    # Enterprise Linux distribution assets (systemd, deb, rpm)
├── pkg/
│   ├── downloader/               # Hugging Face Hub GGUF streaming downloader
│   ├── gguf/                     # GGUF v2/v3 parser, metadata reader, mmap loader
│   ├── quant/                    # Direct Q2_K, Q3_K, Q4_0, Q4_K, Q6_K, Q8_0 dot-products
│   ├── math/                     # RMSNorm, RoPE, SwiGLU (SiLU), Softmax & multithreaded GEMV
│   ├── metal/                    # Apple Metal GPU compute backend & MSL shaders
│   ├── distributed/              # Pipeline Parallelism, Tensor Parallelism & Speculative Decoding
│   ├── tokenizer/                # BPE tokenizer & merge evaluation
│   ├── engine/                   # Transformer forward pass, Prefix Cache, Paged KV & Embeddings
│   ├── sampler/                  # Sampler & JSON grammar constraint masking
│   └── server/                   # HTTP Server (OpenAI, Ollama, Web UI, Metrics)
│       └── web/
│           └── ui.html           # Embedded dark-mode streaming Web UI
```

---

## Prerequisites

- **Go**: Version 1.20 or higher (Go 1.22+ recommended).
- **C Compiler**: **None required!** The engine is 100% pure Go standard library with zero CGo dependencies.

---

## Build Instructions

### Quick Build (via Make)

If you have `make` installed:

```bash
make build        # Build local goinfer binary
make build-prod   # Build optimized binary with stripped symbols
make docker-build # Build Docker container image (go-infer:latest)
make test         # Run test suite
make bench        # Build benchmark tool
make quantize     # Build model quantizer tool
make packages     # Build Debian/Ubuntu (.deb) and RHEL/CentOS (.rpm) packages into bin/dist/
make release      # Cross-compile for Darwin, Linux & Windows into bin/
make install      # Install into $GOPATH/bin
```

---

## Linux Distribution Packaging (Ubuntu/Debian & RHEL/CentOS)

`GoInfer` includes built-in packaging for enterprise Linux distributions with `systemd` service integration, sandboxing, and configuration management.

### 1. Build `.deb` and `.rpm` Packages
```bash
make packages
```
This produces ready-to-install packages in `bin/dist/`:
* `goinfer_0.1_amd64.deb` (Ubuntu / Debian x86_64)
* `goinfer_0.1_arm64.deb` (Ubuntu / Debian ARM64 / Graviton)
* `goinfer-0.1-1.x86_64.rpm` (RHEL / CentOS / Rocky Linux / Fedora)
* `goinfer-0.1-1.aarch64.rpm` (RHEL / Fedora ARM64)

### 2. Install on Ubuntu / Debian
```bash
sudo dpkg -i bin/dist/goinfer_0.1_amd64.deb
```

### 3. Install on RHEL / CentOS / Rocky Linux / Fedora
```bash
sudo rpm -ivh bin/dist/goinfer-0.1-1.x86_64.rpm
```

### 4. Manage Systemd Service
Once installed, `goinfer` runs as a sandboxed system service (`goinfer` system user):

```bash
# Configure model path and listen port
sudo nano /etc/goinfer/goinfer.conf

# Start and inspect the service
sudo systemctl start goinfer
sudo systemctl status goinfer

# View live streaming logs
sudo journalctl -u goinfer -f
```

### Manual Go CLI Build

#### 1. Standard Local Build

Clone the repository and build the binary in the current directory:

```bash
git clone https://github.com/your-username/goinfer.git
cd goinfer
go build -o goinfer .
```

#### 2. Production Optimized Build (Smaller Binary & Stripped Symbols)

To produce a smaller stripped binary without DWARF debug tables:

```bash
go build -ldflags="-s -w" -o goinfer .
```

### 3. Cross-Compilation

Since `goinfer` uses zero CGo, you can easily cross-compile for any target operating system and CPU architecture:

```bash
# macOS Apple Silicon (ARM64)
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o goinfer-darwin-arm64 .

# macOS Intel (x86_64)
CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -o goinfer-darwin-amd64 .

# Linux (x86_64 / AMD64)
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o goinfer-linux-amd64 .

# Linux (ARM64 / Graviton / Raspberry Pi)
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o goinfer-linux-arm64 .

# Windows (x86_64)
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o goinfer.exe .
```

### 4. Install into `$GOPATH/bin`

Install the binary directly into your system's Go bin directory for global CLI access:

```bash
go install .
```

---

## 🐳 Docker & Container Deployment

`GoInfer` provides multi-stage, zero-dependency Docker container images running unprivileged with built-in healthchecks:

### 1. Build Docker Image
```bash
docker build -t go-infer:latest .
```

### 2. Run Container with Local Model Mount
Mount your local directory containing `.gguf` files into `/models`:
```bash
docker run -d \
  --name goinfer \
  -p 8080:8080 \
  -v $(pwd)/models:/models \
  go-infer:latest --serve :8080 /models/model.gguf
```

### 3. Docker Compose
Deploy instantly with the included `docker-compose.yml`:
```bash
docker compose up -d
```
Access the streaming Web UI dashboard at `http://localhost:8080`.

---

## Locating Ollama Models

Ollama stores its GGUF weights directly on your disk in blob format:

- **macOS**: `~/.ollama/models/blobs/`
- **Linux**: `/usr/share/ollama/.ollama/models/blobs/` or `~/.ollama/models/blobs/`
- **Windows**: `C:\Users\<username>\.ollama\models\blobs\`

To find the largest GGUF weight file:

```bash
# macOS / Linux
ls -lhS ~/.ollama/models/blobs/ | head -n 5
```

The largest file (e.g. `sha256-1194192cf2a1...`) is your GGUF model file.

---

## Usage

### 1. Direct Hugging Face Model Downloader (`pull`)
Download any GGUF model directly from Hugging Face Hub with live streaming progress:
```bash
# Auto-discovers recommended Q4_K_M GGUF in repository
./goinfer pull unsloth/Llama-3.2-1B-Instruct-GGUF

# Or pull a specific GGUF file directly
./goinfer pull TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf
```

### 2. Direct CLI Prompt Completion
Run inference directly against any GGUF file or Ollama model blob:
```bash
./goinfer models/llama-3.2-1b-instruct.Q4_K_M.gguf "Explain goroutines in Go in two sentences."
```

### 3. Interactive REPL Chat
Omit the prompt argument to start interactive multi-turn chat with automatic prompt template detection (LLaMA 3 / ChatML):
```bash
./goinfer models/llama-3.2-1b-instruct.Q4_K_M.gguf
```

### 4. HTTP Server with Embedded Web Chat UI & OpenAI / Ollama API
Launch the unified server on port 8080:
```bash
./goinfer --serve :8080 models/llama-3.2-1b-instruct.Q4_K_M.gguf
```

#### A. Embedded Dark-Mode Web UI
Open `http://localhost:8080` in any modern web browser to access the built-in streaming chat dashboard with real-time SSE generation and interactive parameter sliders.

#### B. OpenAI Chat Completions API (Streaming & JSON Mode)
```bash
# Standard Streaming Chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a haiku about Go."}],
    "stream": true,
    "temperature": 0.7
  }'

# JSON Schema / Structured Output Constrained Decoding
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Extract name and age: Alice is 30 years old."}],
    "response_format": {"type": "json_object"}
  }'
```

#### C. Embeddings API (`/v1/embeddings`)
Generate L2-normalized dense vector embeddings for RAG or vector database indexing:
```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "High-performance inference engine in pure Go"
  }'
```

#### D. Prometheus Metrics (`/metrics`)
Scrape real-time server telemetry:
```bash
curl http://localhost:8080/metrics
```

#### E. Ollama Compatible API (`/api/generate`)
```bash
curl http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "Why is the sky blue?",
    "stream": true
  }'
```

---

## Distributed Multi-Server Inference

`GoInfer` supports 3 distributed coordination architectures enabled via CLI flags (by default, it runs standalone locally).

### Choosing the Right Distributed Strategy

| Strategy | Recommended Network Interconnect | Primary Use Case | Expected Benefit |
| :--- | :--- | :--- | :--- |
| **Distributed Speculative Decoding** | **Standard LAN / Wi-Fi / 1GbE** | Accelerating a single user's token generation speed | **2.0× – 3.0× faster generation** (60–80+ tok/s) |
| **Pipeline Parallelism (PP)** | **Standard LAN / 1GbE / 10GbE** | Running huge models (70B–405B) that exceed a single machine's RAM | **Fits massive models across nodes** (1 network hop / token) |
| **Tensor Parallelism (TP)** | **Ultra-low latency (NVLink / InfiniBand)** | Dividing matrix bandwidth on high-speed clusters | **1.8× – 2.0× faster GEMV** (requires 80 syncs / token) |
| **Data Parallelism** | **Any Network / Load Balancer** | Serving multiple concurrent users | **Linear request throughput scaling** ($N\times$) |

---

### Architectural Deep-Dive & Network Latency Tradeoffs

#### 1. Distributed Speculative Decoding (`--dist-mode=speculative`)
* **When to use**: When you have multiple machines on standard home/office Ethernet/Wi-Fi and want to make inference **significantly faster for a single stream**.
* **How it works**:
  * **Draft Server (Node 1)** runs a lightweight model (e.g. 1.5B/3B) at **120+ tok/s** and generates $K$ speculative candidate tokens.
  * **Target Server (Node 2)** receives the draft tokens and verifies all of them concurrently in a **single parallel `ForwardBatch` pass** (taking only ~200ms).
* **Why it's fast over LAN**: Candidate tokens are sent in batches rather than paying network latency on every individual token.

```bash
# Node 1: Start lightweight draft model server (e.g. 1.5B/3B model on port 8081)
./goinfer --serve :8081 path/to/draft-model.gguf

# Node 2: Run target model with distributed speculative acceleration
./goinfer --dist-mode speculative \
  --draft-server http://192.168.1.10:8081 \
  --draft-tokens 4 \
  path/to/target-model-35b.gguf \
  "Explain quantum computing in two sentences."
```

---

#### 2. Pipeline Parallelism (`--dist-mode=pipeline`)
* **When to use**: When a model (e.g. Llama-3-70B or 405B) is **too large to fit in the RAM of a single machine**.
* **How it works**: Server 1 loads and computes layers $0 \dots 19$, then transfers the compact hidden activation vector (~14 KB) over HTTP/TCP to Server 2 to compute layers $20 \dots 39$.
* **Network Overhead**: Only **1 network hop per token** (~0.2ms), making it highly practical over standard Gigabit networks.

```bash
# Node 2: Stage 2 (Layers 20-39, Final Output Logits)
./goinfer --serve :8082 \
  --dist-mode pipeline \
  --pipeline-layers 20-39 \
  path/to/model.gguf

# Node 1: Stage 1 (Layers 0-19, Streams activations to Stage 2)
./goinfer --serve :8081 \
  --dist-mode pipeline \
  --pipeline-layers 0-19 \
  --pipeline-next http://192.168.1.12:8082 \
  path/to/model.gguf
```

---

#### 3. Tensor Parallelism (`--dist-mode=tensor-parallel`)
* **When to use**: When nodes are connected via **ultra-low latency cluster interconnects** (NVLink, RoCE, or 400Gbps InfiniBand).
* **How it works**: Weight matrices are partitioned horizontally and vertically across peer ranks. Nodes compute partial dot products concurrently and sum results via `AllReduce`.
* **Network Overhead**: Requires 2 `AllReduce` synchronizations per layer ($2 \times 40\text{ layers} = \mathbf{80\text{ network roundtrips per token}}$). On standard Ethernet, network roundtrip latency will bottleneck throughput; on high-speed fabrics, it doubles memory bandwidth.

```bash
# Peer 0
./goinfer --dist-mode tensor-parallel --tp-rank 0 --tp-peers "http://node1:8080,http://node2:8080" model.gguf "prompt"

# Peer 1
./goinfer --dist-mode tensor-parallel --tp-rank 1 --tp-peers "http://node1:8080,http://node2:8080" model.gguf "prompt"
```

---

## Memory Reduction & Quantization

### 1. Quantized KV-Cache (`--kv-type=f32|q8_0|q4_0`)
Reduces context memory footprint by storing key and value history in quantized format:
* **`f32`** (default): Full 32-bit floating point precision.
* **`q8_0`**: 8-bit quantized KV cache (**2× context RAM reduction**).
* **`q4_0`**: 4-bit quantized KV cache (**4× context RAM reduction**).

```bash
# Run with 4-bit quantized KV-cache
./goinfer --kv-type q4_0 path/to/model.gguf "Your prompt here"
```

### 2. Standalone Model Quantizer (`cmd/quantize`)
Quantizes unquantized (`F16`/`F32`) or 8-bit models into compact `Q4_0` or `Q8_0` files directly in pure Go:

```bash
# Build the quantizer
make quantize

# Convert model to Q4_0 (reducing model file size by ~75%)
./quantize -input llama-3-8b-f16.gguf -output llama-3-8b-q4_0.gguf -type q4_0
```

---

## Performance Benchmarking (Side-by-Side vs. Native Ollama)

`GoInfer` includes a standalone comparative benchmark runner:

```bash
# Build the benchmark tool
make bench

# Run side-by-side benchmark comparison
./bench --max-tokens 20
```

### Benchmark Results (35B Model / 22 GB File on Apple Silicon M-Series)

```
===============================================================
                      BENCHMARK SUMMARY                        
===============================================================
Engine                    | Prefill Latency | Speed (tokens/s)  
---------------------------------------------------------------
GoInfer (Pure Go + Metal) | 216 ms          | 27.5 - 31.6 tok/s
Ollama Native (C++/GPU)   | 240 ms          | 55.2 - 63.1 tok/s
===============================================================
```

---

## Running Automated Tests & Micro-benchmarks

```bash
# Run all unit and integration tests
make test

# Run Go performance micro-benchmarks
make test-bench
```

---

## 📱 Mobile & iOS Deployment (iPhone 15 Pro / 16 Ready)

Because `GoInfer` is built in **Go + Apple Metal**, it runs natively on iOS devices powered by Apple Silicon A-series chips (A16, A17 Pro, A18, A18 Pro):

| Model | Quantization | RAM Required | Target Speed on iPhone A17/A18 |
| :--- | :--- | :--- | :--- |
| **Llama-3.2-1B** | `Q4_0` / `Q4_K` | **~750 MB** | **80 – 110 tok/s** |
| **Llama-3.2-3B** | `Q4_0` / `Q4_K` | **~1.9 GB** | **45 – 65 tok/s** |
| **Llama-3.1-8B** | **`Q2_K` (2-bit)** | **~2.8 GB** | **35 – 45 tok/s** |
| **Llama-3.1-8B** | **`Q3_K` (3-bit)** | **~3.6 GB** | **28 – 35 tok/s** |

### Building for iOS
```bash
# Build standalone iOS arm64 executable (for sideloading / testing)
CGO_ENABLED=1 GOOS=ios GOARCH=arm64 go build -o goinfer-ios .

# Or compile into an iOS .xcframework via gomobile
gomobile bind -target=ios -o GoInfer.xcframework ./pkg/engine ./pkg/server
```

---

## 🛡️ Security & Concurrency Hardening

`GoInfer` implements production-grade security and synchronization protections:

1. **DoS & Memory Exhaustion Protection**: All HTTP POST endpoints are enforced with a strict **10 MB payload limit** via `http.MaxBytesReader`.
2. **Thread-Safe Engine Synchronization**: Internal `sync.Mutex` locks forward passes to prevent race conditions and memory corruption during concurrent HTTP requests.
3. **Compute Hijack & Context Bounds Check**: Enforces hard ceilings on `max_tokens` bounded by model context length (`e.Config.SeqLen`) and automatically truncates oversized prompts safely.
4. **Configurable CORS Headers**: Fully supports CORS origin restrictions (`--cors-origin`) and HTTP `OPTIONS` preflight requests.
5. **System Sandboxing**: Debian and RPM systemd units run under dedicated unprivileged `goinfer` users with `ProtectSystem=full` and `ProtectHome=true`.

---

## ⚙️ Complete CLI Flag Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model <path>` | `""` | Path to GGUF model file or Ollama blob |
| `--prompt <text>` | `""` | Prompt text to generate completion for |
| `--serve <addr>` | `""` | Start HTTP server on address (e.g. `:8080`) |
| `--threads <n>` | `NumCPU` | Number of worker threads for GEMV |
| `--max-tokens <n>` | `256` | Maximum tokens to generate |
| `--temp <f>` | `0.7` | Sampling temperature (0.0 for greedy) |
| `--top-p <f>` | `0.9` | Top-P nucleus sampling cutoff |
| `--top-k <n>` | `40` | Top-K sampling cutoff |
| `--rep-penalty <f>`| `1.1` | Repetition penalty factor |
| `--kv-type <type>` | `f32` | KV-cache storage precision: `f32`, `q8_0`, `q4_0` |
| `--cors-origin <s>`| `*` | Allowed CORS origin header for HTTP API |
| `--dist-mode <m>` | `none` | Distributed mode: `none`, `speculative`, `pipeline`, `tensor-parallel` |
| `--draft-server <u>`| `""` | Target server URL for speculative decoding |
| `--draft-tokens <n>`| `4` | Number of candidate tokens per draft step |
| `--pipeline-layers <s>`| `""` | Layer partition range (e.g. `0-19`) |
| `--pipeline-next <u>`| `""` | Downstream pipeline server URL |
| `--tp-rank <n>` | `0` | Tensor parallelism rank of this worker |
| `--tp-peers <s>` | `""` | Comma-separated peer URLs for AllReduce |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


