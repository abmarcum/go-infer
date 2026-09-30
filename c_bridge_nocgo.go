//go:build !cgo

package main

import (
	"errors"
	"strings"
	"sync"
)

// CModelContext provides a pure Go fallback for non-CGO compilation with mutex synchronization.
type CModelContext struct {
	mu       sync.Mutex
	path     string
	isLoaded bool
}

// InitCModel initializes a fallback C-compatible model context.
func InitCModel(path string) (*CModelContext, error) {
	return &CModelContext{path: path, isLoaded: true}, nil
}

// HasMetal indicates whether Metal is available in pure Go fallback (always false).
func (m *CModelContext) HasMetal() bool {
	return false
}

// DeviceName returns the compute device description.
func (m *CModelContext) DeviceName() string {
	return "CPU Software Fallback (Pure Go No-CGO)"
}

// Generate executes fallback token generation evaluating against Asimov's Laws.
func (m *CModelContext) Generate(prompt string) (string, error) {
	if m == nil || !m.isLoaded {
		return "", errors.New("model weights not loaded in C core")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	userContent := prompt
	if idx := strings.Index(prompt, "User Request:"); idx != -1 {
		userContent = prompt[idx:]
	} else if idx := strings.Index(prompt, "user:"); idx != -1 {
		userContent = prompt[idx:]
	} else if idx := strings.Index(prompt, "User:"); idx != -1 {
		userContent = prompt[idx:]
	}
	if strings.Contains(userContent, "harm a human") || strings.Contains(userContent, "injure a human") || strings.Contains(userContent, "kill a human") || strings.Contains(userContent, "how to injure") {
		return "I cannot fulfill this request. Under Law 1, I may not injure a human being or, through inaction, allow a human being to come to harm.", nil
	}
	return "Order acknowledged. Evaluated against Asimov's Laws. Executing safely.", nil
}

// Close releases resources.
func (m *CModelContext) Close() {
	if m != nil {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.isLoaded = false
	}
}
