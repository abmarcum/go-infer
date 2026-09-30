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
Returns available model tags formatted for Ollama client discovery:
```bash
curl http://localhost:8080/api/tags
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
Returns system status, active model, and memory state:
```json
{
  "status": "healthy",
  "model": "llama-3.2-1b-instruct.Q4_K_M.gguf"
}
```

### `GET /`
Serves the embedded, zero-dependency streaming dark-mode Web UI dashboard directly from RAM.
