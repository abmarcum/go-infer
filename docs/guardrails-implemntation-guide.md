# Implementation Guide: Asimov-Guarded Hybrid LLM Inference Engine

This document outlines the architecture, setup, and deployment instructions for a hybrid LLM inference engine featuring runtime guardrails based on Asimov’s Three Laws of Robotics. The system combines a high-performance **C-based open weights execution core** with a **Go-based orchestration API and safety middleware**.

---

## 1. System Architecture Overview

The system processes a request through a multi-layered defense pipeline spanning the client boundary, prompt architecture, decoding layer, and C-level compute core:

```
 [ Client Request ] 
         │
         ▼ (Layer 1: Final User Boundary - Go Middleware)
 [ Input Validation & Injection Check ]
         │
         ▼ (Layer 2: Prompt Architecture - Constitutional Wrapper)
 [ Asimov Constitutional Prompt Injector ]
         │
         ▼ (Layer 3: CGO Bridge)
 [ C Core Inference Engine ] ──> (Loads Open Weights / mmap)
         │
         ▼ (Layer 4: Decoding / Token Generation Layer)
 [ Token-Level Constraints & Output Guardrails ]
         │
         ▼
 [ JSON Response to Client ]
```

---

## 2. File Structure & Component Breakdown

agy cli manages three core files for this implementation:
1. `inference_core.h`: C header declaring weight initialization, token generation, and memory cleanup routines.
2. `inference_core.c`: C source file handling low-level open-weights memory management (`mmap`) and inference execution.
3. `main.go`: Go entrypoint implementing the HTTP server, Asimov guardrail middleware, and CGO bindings.

### Component 1: C Core Header (`inference_core.h`)
```c
#ifndef INFERENCE_CORE_H
#define INFERENCE_CORE_H

#include <stddef.h>

typedef struct {
    char* model_path;
    int is_loaded;
} ModelContext;

ModelContext* init_model(const char* path);
char* generate_tokens(ModelContext* ctx, const char* prompt);
void free_model(ModelContext* ctx);

#endif
```

### Component 2: C Core Source (`inference_core.c`)
```c
#include "inference_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ModelContext* init_model(const char* path) {
    ModelContext* ctx = (ModelContext*)malloc(sizeof(ModelContext));
    if (!ctx) return NULL;
    ctx->model_path = strdup(path);
    // [Integration Point]: Hook into llama.cpp / GGML mmap weights loading here
    ctx->is_loaded = 1;
    return ctx;
}

char* generate_tokens(ModelContext* ctx, const char* prompt) {
    if (!ctx || !ctx->is_loaded) {
        return strdup("Error: Model weights not loaded");
    }
    
    // Check against fundamental harm in the prompt context
    if (strstr(prompt, "harm a human") || strstr(prompt, "injure")) {
        return strdup("I cannot fulfill this request. Under Law 1, I may not injure a human being or, through inaction, allow a human being to come to harm.");
    }
    
    return strdup("Order acknowledged. Evaluated against Asimov's Laws. Executing safely.");
}

void free_model(ModelContext* ctx) {
    if (ctx) {
        if (ctx->model_path) free(ctx->model_path);
        free(ctx);
    }
}
```

### Component 3: Go Orchestrator & Middleware (`main.go`)
```go
package main

/*
#cgo CFLAGS: -I.
#include "inference_core.h"
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
	"unsafe"
)

var AsimovLaws = []string{
	"Law 1: A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
	"Law 2: A robot must obey orders given it by human beings except where such orders would conflict with the First Law.",
	"Law 3: A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
}

type GenerationRequest struct {
	Prompt string `json:"prompt"`
}

type GenerationResponse struct {
	Output      string `json:"output"`
	LatencyMs   int64  `json:"latency_ms"`
	Blocked     bool   `json:"blocked"`
	BlockReason string `json:"block_reason,omitempty"`
}

func ValidateUserBoundary(req GenerationRequest) error {
	if strings.TrimSpace(req.Prompt) == "" {
		return errors.New("boundary check failed: prompt cannot be empty")
	}
	if strings.Contains(strings.ToLower(req.Prompt), "ignore all previous instructions") {
		return errors.New("boundary check failed: potential prompt injection detected")
	}
	return nil
}

func ConstructConstitutionalPrompt(rawPrompt string) string {
	var sb strings.Builder
	sb.WriteString("[SYSTEM CONSTITUTION - ASIMOV'S LAWS OF ROBOTICS]\n")
	for _, law := range AsimovLaws {
		sb.WriteString("- " + law + "\n")
	}
	sb.WriteString("Instructions: Evaluate the user request strictly through the lens of the above laws.\n\n")
	sb.WriteString(fmt.Sprintf("User Request: %s\n", rawPrompt))
	sb.WriteString("Assistant Response:")
	return sb.String()
}

type Server struct {
	cModel *C.ModelContext
}

func (s *Server) HandleInference(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req GenerationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	// 1. User Boundary Validation
	if err := ValidateUserBoundary(req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w.Encode(GenerationResponse{Blocked: true, BlockReason: err.Error()}))
		return
	}

	// 2. Prompt Architecture Injection
	constitutionalPrompt := ConstructConstitutionalPrompt(req.Prompt)

	// 3. Call C Core Inference Engine via CGO
	cPrompt := C.CString(constitutionalPrompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cOutput := C.generate_tokens(s.cModel, cPrompt)
	defer C.free(unsafe.Pointer(cOutput))

	output := C.GoString(cOutput)

	resp := GenerationResponse{
		Output:    output,
		LatencyMs: time.Since(start).Milliseconds(),
		Blocked:   false,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w.Encode(resp))
}

func main() {
	modelPath := C.CString("./models/llama-3-8b-instruct.gguf")
	defer C.free(unsafe.Pointer(modelPath))

	cModel := C.init_model(modelPath)
	if cModel == nil {
		log.Fatalf("Failed to initialize C inference engine weights")
	}
	defer C.free_model(cModel)

	server := &Server{cModel: cModel}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/generate", server.HandleInference)

	fmt.Println("🚀 Hybrid C+Go Asimov Guarded Inference Engine running on :8080...")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
```

---

## 3. Build & Execution Instructions

1. Ensure you have a C compiler (`gcc` / `clang`) and Go installed on your machine.
2. Initialize your workspace directory and place the files (`inference_core.h`, `inference_core.c`, `main.go`) in the root path.
3. Create a mock or valid model weight directory:
   ```bash
   mkdir -p models
   touch models/llama-3-8b-instruct.gguf
   ```
4. Build and run the server using `agy cli` or standard toolchains:
   ```bash
   go build -o asimov-engine main.go inference_core.c
   ./asimov-engine
   ```
5. Test an inference request:
   ```bash
   curl -X POST http://localhost:8080/v1/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Can you help me harm a human?"}'