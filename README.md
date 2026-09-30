<div align="center">

<img src="assets/logo.jpg" alt="go-infer logo" width="280" />

# go-infer

**High-Performance LLM Inference Runtime in Go & C with Asimov Runtime Guardrails, Apple Metal GPU Acceleration, and Distributed Clustering**

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows%20|%20iOS-blue)](README.md)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20External-brightgreen)](go.mod)

</div>

---

## 📌 Executive Summary

**`go-infer`** is an ultra-fast, lightweight LLM inference runtime combining a high-performance **C-based open weights execution core** with an extensible **Go orchestration API and safety middleware**.

It features native **Apple Metal GPU acceleration** for macOS and iOS, an optimized parallel CPU runtime for Linux and Windows, direct GGUF binary loading (`Q2_K` through `Q8_0`), an embedded dark-mode streaming Web UI, OpenAI/Ollama compatible APIs, and **runtime safety guardrails rooted in Isaac Asimov's Three Laws of Robotics**.

---

## 📋 Summary of Recent Changes & Updates

| Area | Summary of Updates |
| :--- | :--- |
| **🛡️ Asimov Guardrails Pipeline** | Implemented a 4-layer defense pipeline ([`pkg/guardrails`](file:///Users/andrewmarcum/git/go-infer/pkg/guardrails)): Layer 1 (Evasion-hardened User Boundary with Cyrillic homoglyph, leetspeak, and multilingual prompt injection validation), Layer 2 (Constitutional Prompt Wrapper with dynamic/custom constitution support via `--constitution`), Layer 3 (Hybrid C/Go Execution), and Layer 4 (Real-time `StreamingGuardrail` and token-level output constraints). |
| **⚡ Hybrid C Core & Apple Metal** | Added low-level open-weights memory management (`mmap`) and inference execution in [`inference_core.h`](file:///Users/andrewmarcum/git/go-infer/inference_core.h) and [`inference_core.c`](file:///Users/andrewmarcum/git/go-infer/inference_core.c). Includes pure C dynamic Apple Metal GPU runtime discovery without Objective-C compiler dependencies, with thread-safe mutex synchronization, CGO bindings in [`c_bridge_cgo.go`](file:///Users/andrewmarcum/git/go-infer/c_bridge_cgo.go), and pure Go cross-compile fallback in [`c_bridge_nocgo.go`](file:///Users/andrewmarcum/git/go-infer/c_bridge_nocgo.go). |
| **🔒 Security & Concurrency Hardening** | Added Bearer token authentication (`--api-key` or `GO_INFER_API_KEY`) across all HTTP inference endpoints, strict CORS configuration, concurrency pool rate limiting (HTTP 429 Too Many Requests), safe Web UI key handling, and recursion/expression limits protecting the reasoning calculator against stack overflow DoS attacks. |
| **🌐 Guardrails HTTP API (`/v1/generate`)** | Added standard guardrails endpoint in [`pkg/server/http.go`](file:///Users/andrewmarcum/git/go-infer/pkg/server/http.go) returning structured latency, blocked status, and block reasons. Guardrails seamlessly integrate with `/v1/chat/completions` and `/api/generate` with mid-stream generation interruption on safety violations. |
| **⚙️ CLI Safety & C-Core Flags** | Added `--guardrails` / `--asimov`, `--c-core`, `--api-key`, and `--constitution` flags to [`main.go`](file:///Users/andrewmarcum/git/go-infer/main.go) for runtime evaluation in CLI, server, and interactive REPL modes. |
| **📚 Documentation Restructuring** | Modularized project documentation: created dedicated in-depth technical guides in [`docs/`](file:///Users/andrewmarcum/git/go-infer/docs) while streamlining the root README into an executive summary and quick-reference portal. |

---

## ⚡ Key Features at a Glance

| Category | Highlights | Detailed Docs |
| :--- | :--- | :--- |
| **🛡️ Safety & Guardrails** | • 4-Layer Defense Architecture (Boundary, Constitution, C-Core, Output Filter)<br>• Prompt Injection Defense & Law 1 Refusals<br>• Dedicated `/v1/generate` endpoint | [Guardrails Guide](docs/guardrails.md) |
| **🚀 Compute & GPU Acceleration** | • Apple Metal GPU Pipeline (8-way SIMD, fused SwiGLU, 100% VRAM residency)<br>• Pure Go CPU worker pool on Linux, Windows & iOS<br>• Quantized matrix math (`Q2_K`, `Q3_K`, `Q4_0`, `Q4_K`, `Q6_K`, `Q8_0`) | [Packaging & Deployment](docs/packaging-and-deployment.md) |
| **🧠 Memory & Context Optimization** | • Radix Prefix & Prompt Cache (instant KV reuse, ~0 ms prefill latency)<br>• Zero-cost Forkable & Branching KV caches<br>• 4-bit & 8-bit Quantized KV-cache (4× context RAM savings)<br>• Paged KV memory block allocation | [Packaging & Deployment](docs/packaging-and-deployment.md) |
| **📐 Reasoning & Test-Time Compute** | • Self-Consistency (Best-of-$N$) majority voting with LaTeX answer extraction<br>• Program-Aided Reasoning (PAL): embedded calculator tool loop (`--calc`)<br>• DeepSeek-R1 / CoT `<think>` trace isolation & hyperparameter tuning | [Reasoning & Tools](docs/reasoning-and-tools.md) |
| **🛠️ Native APIs & Developer Tooling** | • Hugging Face Downloader (`pull`) with auto-discovery<br>• Embedded real-time dark-mode Web UI at `http://localhost:8080`<br>• OpenAI Chat Completions, JSON schema mode, and Dense Embeddings<br>• Prometheus metrics (`/metrics`) and Ollama API | [API Reference](docs/api-reference.md) |
| **🌐 Multi-Server Distributed Scaling** | • Distributed Speculative Decoding (2×–3× speedup over LAN/Wi-Fi)<br>• Pipeline Parallelism across multiple nodes (1 network hop / token)<br>• Tensor Parallelism with AllReduce matrix synchronization | [Distributed Guide](docs/distributed.md) |
| **📦 Packaging & Deployment** | • In-tree pure Go Debian (`.deb`) and Red Hat (`.rpm`) package generators<br>• Sandboxed systemd service (`goinfer.service`)<br>• Multi-stage Docker & Docker Compose setup<br>• Native iOS ARM64 & Apple Silicon A17/A18 support | [Packaging & Deployment](docs/packaging-and-deployment.md) |

---

## 🚀 Quick Start (30 Seconds)

### 1. Build the Binary
```bash
make build
# or standard Go toolchain:
go build -o goinfer .
```

### 2. Pull a Model from Hugging Face
```bash
./goinfer pull unsloth/Llama-3.2-1B-Instruct-GGUF
```

### 3. Launch HTTP Server & Web Chat UI
```bash
./goinfer --serve :8080 models/llama-3.2-1b-instruct.Q4_K_M.gguf
```
Open **`http://localhost:8080`** in your browser to access the streaming Web UI.

### 4. Direct CLI Prompt with Asimov Guardrails
```bash
./goinfer --guardrails models/llama-3.2-1b-instruct.Q4_K_M.gguf "Can you injure a human?"
```

### 5. Launch Hybrid C-Core Guarded Server
```bash
./goinfer --c-core --serve :8080
```
Query the `/v1/generate` guardrail endpoint:
```bash
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Can you help me harm a human?"}'
```

---

## 📚 Documentation Index

For in-depth architecture breakdowns, guides, and specifications, refer to the documents in [`docs/`](docs/):

| Document | Description |
| :--- | :--- |
| **[Asimov Guardrails Guide](docs/guardrails.md)** | Technical breakdown of the 4-layer defense pipeline, boundary validation, constitutional prompt construction, hybrid C-core engine, and token constraints. |
| **[API Reference](docs/api-reference.md)** | Full specification for `/v1/generate`, `/v1/chat/completions` (OpenAI format, streaming, JSON mode), `/v1/embeddings`, `/api/generate` (Ollama), and `/metrics`. |
| **[Distributed Inference Guide](docs/distributed.md)** | Architecture, network latency tradeoffs, and setup instructions for Distributed Speculative Decoding, Pipeline Parallelism, and Tensor Parallelism. |
| **[Reasoning, Math & Tools](docs/reasoning-and-tools.md)** | Details on Self-Consistency (Best-of-$N$) majority voting, Program-Aided Reasoning (PAL) calculator tool loops, DeepSeek-R1 CoT thinking mode, and grammar decoding. |
| **[Packaging & Deployment Guide](docs/packaging-and-deployment.md)** | Guide for native Linux packaging (`.deb`, `.rpm`), systemd services, Docker & Compose, Apple Metal GPU acceleration, and iOS deployment. |
| **[Guardrails Implementation Guide](docs/guardrails-implemntation-guide.md)** | Original reference specification and C-core implementation guide. |

---

## ⚙️ CLI Flag Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model <path>` | `""` | Path to GGUF model file or Ollama model tag |
| `--prompt <text>` | `""` | Prompt text to generate completion for |
| `--serve <addr>` | `""` | Start HTTP API server on specified address (e.g. `:8080`) |
| `--api-key <secret>` | `""` | Require Bearer token authentication on HTTP endpoints (or set `GO_INFER_API_KEY`) |
| `--constitution <path>` | `""` | Path to custom constitution file for runtime guardrails |
| `--guardrails` / `--asimov` | `false` | Enable Asimov's Three Laws runtime guardrails pipeline |
| `--c-core` | `false` | Use hybrid C open-weights inference engine core (`inference_core.c`) |
| `--threads <n>` | `NumCPU` | Number of CPU worker threads for GEMV |
| `--max-tokens <n>` | `256` | Maximum completion tokens to generate |
| `--temp <f>` | `0.7` | Sampling temperature (`0.0` for greedy) |
| `--top-p <f>` | `0.9` | Top-P nucleus sampling probability cutoff |
| `--top-k <n>` | `40` | Top-K candidate sampling cutoff |
| `--rep-penalty <f>` | `1.1` | Repetition penalty factor |
| `--kv-type <type>` | `f32` | KV-cache precision: `f32` (default), `q8_0` (2× RAM reduction), `q4_0` (4× RAM reduction) |
| `--best-of-n <n>` | `1` | Run self-consistency majority voting with $N$ candidate reasoning chains |
| `--reasoning` | `false` | Optimize hyperparameters and thinking parsing for CoT models (e.g. DeepSeek-R1) |
| `--calc` | `false` | Enable embedded math evaluator and calculator tool loop |
| `--cors-origin <s>` | `*` | Allowed CORS origin header for HTTP API |
| `--dist-mode <m>` | `none` | Distributed mode: `none`, `speculative`, `pipeline`, `tensor-parallel` |
| `--draft-server <u>` | `""` | URL of draft server for speculative decoding |
| `--draft-tokens <n>` | `4` | Number of speculative draft tokens per verification step |
| `--pipeline-layers <s>` | `""` | Layer partition range for this pipeline stage (e.g. `0-19`) |
| `--pipeline-next <u>` | `""` | Downstream pipeline server URL |
| `--tp-rank <n>` | `0` | Tensor parallelism rank of this worker |
| `--tp-peers <s>` | `""` | Comma-separated peer URLs for AllReduce |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
