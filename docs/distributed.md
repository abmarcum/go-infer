# Distributed Multi-Server Inference

`go-infer` supports three distributed coordination architectures designed to overcome memory capacity limits and accelerate generation throughput across multiple machines.

---

## 1. Choosing the Right Architecture

| Architecture | Recommended Network | Primary Use Case | Expected Benefit |
| :--- | :--- | :--- | :--- |
| **Distributed Speculative Decoding** | **Standard LAN / Wi-Fi / 1GbE** | Accelerating a single user's token generation stream | **2.0× – 3.0× faster generation** (60–90+ tok/s) |
| **Pipeline Parallelism (PP)** | **Standard LAN / 1GbE / 10GbE** | Running massive models (70B–405B) exceeding single-host RAM | **Runs huge models across nodes** (1 network hop / token) |
| **Tensor Parallelism (TP)** | **Ultra-low latency (InfiniBand / NVLink)** | Dividing matrix bandwidth on high-speed compute clusters | **1.8× – 2.0× faster GEMV** (requires 80 syncs / token) |

---

## 2. Distributed Speculative Decoding (`--dist-mode=speculative`)

### Architecture
Speculative decoding uses a lightweight "draft" model running on one machine to propose $K$ candidate tokens, while a larger "target" model on a second machine verifies all $K$ candidates in a single parallel forward pass.

```
 [ Node 1: Draft Server (1.5B/3B) ] 
          │  Generates K candidate tokens (120+ tok/s)
          ▼
 [ HTTP POST /v1/dist/speculative-draft ]
          │
          ▼
 [ Node 2: Target Server (8B/70B) ]
          │  Single parallel verification pass (ForwardBatch)
          ▼  Accepts matching tokens; resamples on divergence
 [ Emits verified tokens to client ]
```

### Why It Excels Over Standard LAN/Wi-Fi
Traditional distributed inference requires synchronizing intermediate tensor activations on every single transformer layer or token. Speculative decoding batches proposals over HTTP, requiring only one round-trip every $K$ tokens.

### Configuration & Deployment
```bash
# 1. On Node 1 (Draft Worker): Run draft model server on port 8081
./goinfer --serve :8081 models/llama-3.2-1b-instruct.Q4_K_M.gguf

# 2. On Node 2 (Target Worker): Run target model with speculative acceleration
./goinfer --dist-mode speculative \
  --draft-server http://192.168.1.10:8081 \
  --draft-tokens 4 \
  models/llama-3.1-8b-instruct.Q4_K_M.gguf \
  "Explain general relativity in three sentences."
```

---

## 3. Pipeline Parallelism (`--dist-mode=pipeline`)

### Architecture
Splits the layers of a large neural network sequentially across multiple machines connected over TCP/HTTP.

```
 [ Client Request ] 
         │
         ▼
 [ Node 1: Stage 1 (Layers 0-19) ] 
         │  Streams intermediate hidden activation (~14 KB)
         ▼  (POST /v1/dist/pipeline-forward)
 [ Node 2: Stage 2 (Layers 20-39) ] 
         │  Computes final layers, norm, & output logits
         ▼
 [ Emits generated token ]
```

### Network Overhead
Only **1 network hop per token** (~0.2ms over Gigabit LAN), transferring a small activation vector rather than large weight matrices.

### Configuration & Deployment
```bash
# 1. On Node 2 (Stage 2 - Final Layers):
./goinfer --serve :8082 \
  --dist-mode pipeline \
  --pipeline-layers 20-39 \
  models/llama-3-70b-instruct.Q4_K_M.gguf

# 2. On Node 1 (Stage 1 - Initial Layers):
./goinfer --serve :8081 \
  --dist-mode pipeline \
  --pipeline-layers 0-19 \
  --pipeline-next http://192.168.1.12:8082 \
  models/llama-3-70b-instruct.Q4_K_M.gguf
```

---

## 4. Tensor Parallelism (`--dist-mode=tensor-parallel`)

### Architecture
Horizontally and vertically shards attention and feed-forward weight matrices ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) across peer workers. Ranks compute partial dot products concurrently and sum results via `AllReduce`.

### Network Requirements
Requires 2 `AllReduce` synchronizations per transformer layer (e.g. 80 network round-trips per generated token for a 40-layer model). Recommended strictly for ultra-low latency cluster interconnects (InfiniBand / RoCE).

### Configuration & Deployment
```bash
# Worker Rank 0:
./goinfer --dist-mode tensor-parallel \
  --tp-rank 0 \
  --tp-peers "http://10.0.0.1:8080,http://10.0.0.2:8080" \
  models/llama-3-70b.gguf "Your prompt"

# Worker Rank 1:
./goinfer --dist-mode tensor-parallel \
  --tp-rank 1 \
  --tp-peers "http://10.0.0.1:8080,http://10.0.0.2:8080" \
  models/llama-3-70b.gguf "Your prompt"
```
