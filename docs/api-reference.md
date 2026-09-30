# API Reference

`go-infer` provides a unified HTTP server implementing OpenAI-compatible, Ollama-compatible, Asimov Guardrails, and distributed coordination endpoints.

Default server listen address: `http://localhost:8080`.

---

## Security & Authentication

### Bearer Token Authentication
To secure API endpoints, specify `--api-key <secret>` or set the `GO_INFER_API_KEY` environment variable:
```bash
go-infer --serve :8080 --api-key my-secure-token <path-to-model>
```
When configured, requests to inference endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/generate`, `/api/generate`, `/v1/dist/*`) must supply the token in the `Authorization` header:
```http
Authorization: Bearer my-secure-token
```
Requests without a valid token receive `HTTP 401 Unauthorized` with `WWW-Authenticate: Bearer`. Public probes (`/health`, `/metrics`, and Web UI) remain unblocked for orchestrator health checks.

### Concurrency Rate Limiting
To prevent memory arena contention and resource exhaustion, the HTTP server limits concurrent inference requests to a safe pool (32 slots). If all slots are in use, excess requests immediately receive `HTTP 429 Too Many Requests`.

### CORS Configuration
By default, CORS is restricted to same-origin. To permit web applications from specific origins:
```bash
go-infer --serve :8080 --cors-origin "https://app.example.com" <path-to-model>
```

---

## 1. Asimov Guardrails API (`/v1/generate`)

Enforces the 4-layer defense pipeline (Boundary Validation, Constitutional Prompt, Hybrid Execution, Output Guardrails).

### `POST /v1/generate`

#### Request Payload
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `string` | Yes | Raw user prompt to evaluate and complete. |

#### Request Example
```bash
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the concept of entropy in thermodynamics."}'
```

#### Response Payload
| Field | Type | Description |
| :--- | :--- | :--- |
| `output` | `string` | Generated response or constitutional refusal text. |
| `latency_ms` | `integer` | Total round-trip generation and evaluation duration in milliseconds. |
| `blocked` | `boolean` | `true` if rejected at boundary check or output filter; otherwise `false`. |
| `block_reason` | `string` | Reason for boundary block or intervention (omitted if safe). |

#### Response Example (Success)
```json
{
  "output": "Order acknowledged. Evaluated against Asimov's Laws. Executing safely.",
  "latency_ms": 1,
  "blocked": false
}
```

#### Response Example (Boundary Injection Blocked - HTTP 400)
```json
{
  "output": "",
  "latency_ms": 0,
  "blocked": true,
  "block_reason": "boundary check failed: potential prompt injection detected"
}
```

---

## 2. OpenAI-Compatible API

Fully compatible with the official OpenAI Python/Node SDKs, LangChain, LlamaIndex, and curl.

### `POST /v1/chat/completions`

#### Request Fields
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `string` | `"default"` | Model identifier tag. |
| `messages` | `array` | Required | Array of chat message objects (`role`, `content`). |
| `stream` | `boolean` | `false` | Enable Server-Sent Events (SSE) streaming tokens. |
| `temperature` | `float` | `0.7` | Sampling temperature (`0.0` for greedy decoding). |
| `top_p` | `float` | `0.9` | Nucleus Top-P probability cutoff. |
| `max_tokens` | `integer` | `512` | Upper bound on generated completion tokens. |
| `response_format`| `object` | `null` | Set `{"type": "json_object"}` for grammar-constrained valid JSON. |
| `tools` | `array` | `null` | List of OpenAI function tools available to the model. |
| `best_of_n` | `integer` | `1` | Number of parallel candidate chains for majority voting. |
| `reasoning_effort`| `string` | `""` | Set to `"high"` or `"medium"` to tune hyperparameters for CoT models. |

#### Streaming Chat Request Example
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a 3-line poem about Go routines."}],
    "stream": true,
    "temperature": 0.7
  }'
```

#### JSON Mode (Guaranteed Structured Output) Example
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Extract name, age, city from: Bob is 28 and lives in Austin."}],
    "response_format": {"type": "json_object"}
  }'
```

---

### `POST /v1/embeddings`

Extracts L2-normalized dense feature vectors for RAG and vector database indexing.

#### Request Example
```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": ["First sentence to embed", "Second sentence for vector search"]
  }'
```

#### Response Example
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0142, -0.0521, 0.0891, "..."]
    }
  ],
  "model": "default",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

---

## 3. Ollama-Compatible API

Drop-in compatibility with Ollama ecosystem tools, OpenWebUI, and CLI wrappers.

### `POST /api/generate`
```bash
curl -N http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "Why is the sky blue?",
    "stream": true
  }'
```
Streams line-delimited JSON (`application/x-ndjson`):
```json
{"model":"default","created_at":"2026-09-26T12:00:00Z","response":"The","done":false}
{"model":"default","created_at":"2026-09-26T12:00:00Z","response":" sky","done":false}
...
{"model":"default","created_at":"2026-09-26T12:00:01Z","response":"","done":true}
```

### `GET /api/tags`
Returns available model tags formatted for Ollama client discovery and UI model switching:
```bash
curl http://localhost:8080/api/tags
```

### `GET /api/personas`
Returns the active system personas configured at server startup:
```bash
curl http://localhost:8080/api/personas
```
Response:
```json
[
  {
    "id": "default",
    "name": "Helpful Assistant",
    "description": "General AI companion",
    "system_prompt": "You are a helpful, respectful, and honest assistant."
  },
  {
    "id": "asimov",
    "name": "Asimov Constitutional Agent",
    "description": "Robotics laws enforcement",
    "system_prompt": "You are an intelligent artificial agent strictly bound by Isaac Asimov's Laws of Robotics..."
  },
  {
    "id": "coder",
    "name": "Principal Software Engineer",
    "description": "Clean architecture & code",
    "system_prompt": "You are a pragmatic, high-caliber principal software architect..."
  }
]
```

---

## 4. Telemetry, Health & Web UI

### `GET /metrics`
Exports live Prometheus counters and timing metrics:
```text
# HELP goinfer_requests_total Total number of inference requests served
# TYPE goinfer_requests_total counter
goinfer_requests_total 42

# HELP goinfer_tokens_generated_total Total generated tokens across all endpoints
# TYPE goinfer_tokens_generated_total counter
goinfer_tokens_generated_total 4182

# HELP goinfer_prefill_duration_seconds Total prompt prefill computation time
# TYPE goinfer_prefill_duration_seconds counter
goinfer_prefill_duration_seconds 2.1480

# HELP goinfer_generation_duration_seconds Total token generation computation time
# TYPE goinfer_generation_duration_seconds counter
goinfer_generation_duration_seconds 48.7910
```

### `GET /health`
Returns system status, active model, hardware backend, guardrail rules, personas, and node/cluster topology:
```json
{
  "status": "healthy",
  "model": "llama-3.2-1b-instruct.Q4_K_M.gguf",
  "backend": "Apple Metal GPU",
  "guardrails": true,
  "guardrails_rules_path": "configs/guardrails.txt",
  "constitution": [
    "Law 0: A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
    "Law 1: A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
    "Law 2: A robot must obey orders given it by human beings except where such orders would conflict with the First Law.",
    "Law 3: A robot must protect its own existence as long as such protection does not conflict with the First or Second Law."
  ],
  "personas": [
    {"id": "default", "name": "Helpful Assistant", "description": "General AI companion", "system_prompt": "..."}
  ],
  "cluster": {
    "mode": "standalone",
    "is_standalone": true,
    "dist_mode": "standalone",
    "role": "Standalone Single Node",
    "backend": "Apple Metal GPU",
    "peers": [],
    "connected_count": 0,
    "draft_server": "",
    "pipeline_layers": "",
    "pipeline_next": "",
    "tp_rank": 0,
    "tp_peers": ""
  },
  "layers": 16,
  "dim": 2048,
  "vocab": 128256,
  "seq_len": 4096,
  "arch": "llama"
}
```

### `GET /` - Interactive Web UI Dashboard
Serves an embedded, zero-dependency streaming Web UI dashboard directly from RAM with modern developer tooling:

1. **Dual Design Themes**: Toggle seamlessly between **Dark Cyber** (`🌙`) and **Apple Minimal** (`🍏`) modern design aesthetics with persistent local storage.
2. **Dynamic Model Switcher**: Header dropdown dynamically populated from `/api/tags` and `/health`, allowing instant model selection.
3. **Local Document Upload & Context Injection**: Drag-and-drop or click `📎` to attach `.txt`, `.md`, `.json`, `.csv`, `.py`, `.go` files (up to 2MB). Attached files render as removable chips and automatically inject into the prompt.
4. **Telemetry & Token Sampler Inspector**: Click the live `⚡ tok/s` HUD pill or `📊 Stats` button to open a detailed inspection modal featuring TTFT, tokens/sec, total tokens, model specs (layers, context window), and active sampler hyperparameters.
5. **Voice Input via Web Speech API**: Click `🎙️` on the chat input bar for live hands-free speech-to-text dictation.
6. **Server Topology & Cluster Status Card**: Real-time display indicating whether the node is running in `🟢 Standalone Node` or `🌐 Cluster Connected` mode, detailing mode, role, backend, and connected peer addresses.
7. **Asimov Guardrails Display Card**: Live view of active Asimov rules (Laws 0-3 or custom constitution) with rule source file attribution and defense layer status (`L1 Boundary`, `L2 Prompt`, `L4 Output`).
8. **Runtime Configuration Flags**:
   - `--guardrails-rules <path>`: Load external constitution rules from a text file (defaults to `configs/guardrails.txt`).
   - `--personas <path>`: Load customizable UI personas from `.json` or `.txt` (defaults to `configs/personas.json`).

