# Reasoning, Mathematics & Test-Time Compute

`go-infer` includes native inference-time compute algorithms and embedded tools designed to dramatically increase accuracy on challenging logic, science, and math benchmarks (e.g. GSM8K, MATH).

---

## 1. Self-Consistency / Best-of-$N$ Majority Voting (`--best-of-n`)

### Overview
Instead of relying on a single greedy or sampled completion, Self-Consistency samples $N$ diverse reasoning paths from the model, extracts normalized final answers (including LaTeX `\boxed{...}` or trailing numbers), and determines the consensus winner via majority voting.

### Key Performance Feature: Zero-Cost KV Branching
Unlike naive implementations that re-evaluate the full prompt $N$ times, `go-infer` prefills the prompt once into an immutable prefix KV cache, snapshots the state, and branches $N$ generation paths independently without duplicate compute.

### CLI Usage Example
```bash
./goinfer --best-of-n 5 models/deepseek-r1-distill-qwen-1.5b.gguf \
  "Janet’s ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at the farmers' market for $2 per egg. How much in dollars does she make daily?"
```

### Output Example
```text
--- Self-Consistency Voting (5 paths) ---
>>> Candidate 1:
Answer: She sells 9 eggs at $2 each, making \boxed{18}. (Extracted: 18)
...
═════════════════════════════════════════════════════
🏆 Consensus Majority Answer: 18
📊 Confidence: 100.0% (5/5 votes)
⚡ Throughput: 74.20 tok/s across 612 tokens
═════════════════════════════════════════════════════
```

---

## 2. Embedded Math Evaluator & Calculator Tool Loop (`--calc`)

### Overview
Large language models frequently hallucinate intermediate arithmetic calculations. `go-infer` provides an embedded, pure Go recursive-descent math parser and evaluator directly inside the token generation loop.

### How It Works
When `--calc` is active, the engine monitors the generation stream:
1. When the model emits a calculation block (e.g. ````calc\n<expression>\n```` or `<<calc: <expression>>>`), generation pauses.
2. The Go math engine evaluates the mathematical expression with arbitrary precision (supporting `+`, `-`, `*`, `/`, `^`, `sqrt`, `sin`, `cos`, `log`, etc.).
3. The verified numeric result is injected directly into the active KV-cache context.
4. Token generation resumes with exact arithmetic grounding.

### Security & Denial-of-Service (DoS) Protection
Implemented in [`pkg/reasoning/calculator.go`](file:///Users/andrewmarcum/git/go-infer/pkg/reasoning/calculator.go) to protect production runtime services:
- **Expression Length Limit (`MaxExprLength = 2048`)**: Rejects payload inflation attacks attempting to stall token generation.
- **AST Recursion Depth Limit (`MaxRecursionDepth = 64`)**: Strictly bounds recursive descent parser depth, preventing stack overflow crashes on deeply nested adversarial parentheses `((((...))))`.
- **Power & Exponent Bounds (`MaxExponent = 1000`)**: Prevents floating-point infinity and NaN lockups from extreme exponents.

### CLI Usage Example
```bash
./goinfer --calc models/llama-3.2-1b-instruct.Q4_K_M.gguf \
  "What is 3^7 * sqrt(144) - 250?"
```

---

## 3. DeepSeek-R1 / CoT Reasoning Mode (`--reasoning`)

### Overview
Models trained on Chain-of-Thought (CoT) and Reinforcement Learning (e.g. DeepSeek-R1, QwQ) require specialized hyperparameter tuning to maintain long thought chains without degradation or repetitive loops.

### Features
- **Hyperparameter Optimization**: Sets $T=0.6$, $\text{Top-P}=0.95$, and disables repetition penalty to preserve mathematical formulas and code syntax.
- **Thinking Trace Separation**: Parses and isolates internal `<think>...</think>` scratchpad tokens from final synthesized responses.
- **Answer Extraction**: Automatically detects and surfaces final answers from the reasoning stream.

### CLI Usage Example
```bash
./goinfer --reasoning models/deepseek-r1-distill-qwen-1.5b.gguf \
  "Prove that the square root of 2 is irrational."
```

---

## 4. Structured JSON Grammar Decoding

### Overview
Enforces guaranteed, schema-valid JSON generation by masking invalid token logits at each decoding step. The model is constrained to emit tokens that strictly adhere to JSON grammar rules, eliminating syntax errors in automated agent workflows.

### Enabling via OpenAI API
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Extract name and score: Alice scored 98."}],
    "response_format": {"type": "json_object"}
  }'
```
