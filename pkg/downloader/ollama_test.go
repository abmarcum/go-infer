package downloader

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModelPathDirectFile(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.gguf")
	if err := os.WriteFile(filePath, []byte("GGUF"), 0644); err != nil {
		t.Fatalf("failed to write dummy file: %v", err)
	}

	resolved, err := ResolveModelPath(filePath)
	if err != nil {
		t.Fatalf("ResolveModelPath failed on direct file: %v", err)
	}
	if resolved != filePath {
		t.Errorf("got %q, want %q", resolved, filePath)
	}
}

func TestResolveModelPathOllamaManifest(t *testing.T) {
	tmpDir := t.TempDir()
	homeDir := filepath.Join(tmpDir, "home")
	t.Setenv("HOME", homeDir)

	modelsDir := filepath.Join(homeDir, ".ollama", "models")
	manifestDir := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "qwen3.6")
	blobsDir := filepath.Join(modelsDir, "blobs")
	if err := os.MkdirAll(manifestDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(blobsDir, 0755); err != nil {
		t.Fatal(err)
	}

	blobPath := filepath.Join(blobsDir, "sha256-1234567890abcdef")
	if err := os.WriteFile(blobPath, []byte("GGUF_DATA"), 0644); err != nil {
		t.Fatal(err)
	}

	mf := ollamaManifest{
		Layers: []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
			Size      int64  `json:"size"`
		}{
			{
				MediaType: "application/vnd.ollama.image.model",
				Digest:    "sha256:1234567890abcdef",
				Size:      1000,
			},
		},
	}
	mfBytes, _ := json.Marshal(mf)
	if err := os.WriteFile(filepath.Join(manifestDir, "27b"), mfBytes, 0644); err != nil {
		t.Fatal(err)
	}

	resolved, err := ResolveModelPath("qwen3.6:27b")
	if err != nil {
		t.Fatalf("ResolveModelPath failed to resolve tag: %v", err)
	}
	if resolved != blobPath {
		t.Errorf("got %q, want %q", resolved, blobPath)
	}
}
