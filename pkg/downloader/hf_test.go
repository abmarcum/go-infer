package downloader

import (
	"testing"
)

func TestResolveHFRepoAndFile(t *testing.T) {
	// 1. Direct repo and file specified
	repo, file, err := resolveHFRepoAndFile("TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf")
	if err != nil {
		t.Fatalf("Failed to parse repo/file: %v", err)
	}
	if repo != "TheBloke/Llama-2-7B-Chat-GGUF" {
		t.Errorf("Expected repo 'TheBloke/Llama-2-7B-Chat-GGUF', got %s", repo)
	}
	if file != "llama-2-7b-chat.Q4_K_M.gguf" {
		t.Errorf("Expected file 'llama-2-7b-chat.Q4_K_M.gguf', got %s", file)
	}

	// 2. Full URL format
	repo, file, err = resolveHFRepoAndFile("https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/llama-3.2-1b-instruct.Q4_K_M.gguf")
	if err != nil {
		t.Fatalf("Failed to parse full URL: %v", err)
	}
	if repo != "unsloth/Llama-3.2-1B-Instruct-GGUF" {
		t.Errorf("Expected repo 'unsloth/Llama-3.2-1B-Instruct-GGUF', got %s", repo)
	}
	if file != "llama-3.2-1b-instruct.Q4_K_M.gguf" {
		t.Errorf("Expected file 'llama-3.2-1b-instruct.Q4_K_M.gguf', got %s", file)
	}
}
