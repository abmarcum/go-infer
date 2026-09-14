package downloader

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

// ResolveModelPath returns the file path if it exists, or resolves an Ollama model tag (e.g. "qwen3.6:27b")
// into its underlying GGUF blob path in ~/.ollama/models/blobs/.
func ResolveModelPath(modelPathOrTag string) (string, error) {
	// 1. Direct file path check
	if info, err := os.Stat(modelPathOrTag); err == nil && !info.IsDir() {
		return modelPathOrTag, nil
	}

	// 2. Check candidate Ollama model roots
	var candidateRoots []string
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		candidateRoots = append(candidateRoots, filepath.Join(home, ".ollama", "models"))
	}
	candidateRoots = append(candidateRoots, "/usr/share/ollama/.ollama/models")

	modelTag := modelPathOrTag
	var name, tag string
	if strings.Contains(modelTag, ":") {
		parts := strings.SplitN(modelTag, ":", 2)
		name = parts[0]
		tag = parts[1]
	} else {
		name = modelTag
		tag = "latest"
	}

	for _, root := range candidateRoots {
		if _, err := os.Stat(root); err != nil {
			continue
		}

		// Look for manifest under standard registry path
		manifestPaths := []string{
			filepath.Join(root, "manifests", "registry.ollama.ai", "library", name, tag),
			filepath.Join(root, "manifests", "registry.ollama.ai", name, tag),
			filepath.Join(root, "manifests", name, tag),
		}

		for _, mp := range manifestPaths {
			data, err := os.ReadFile(mp)
			if err != nil {
				continue
			}

			var mf ollamaManifest
			if err := json.Unmarshal(data, &mf); err != nil {
				continue
			}

			// Find model weight layer
			var modelDigest string
			var maxSize int64
			for _, l := range mf.Layers {
				if l.MediaType == "application/vnd.ollama.image.model" {
					modelDigest = l.Digest
					break
				}
				if l.Size > maxSize {
					maxSize = l.Size
					modelDigest = l.Digest
				}
			}

			if modelDigest != "" {
				blobFileName := strings.Replace(modelDigest, ":", "-", 1)
				blobPath := filepath.Join(root, "blobs", blobFileName)
				if bInfo, err := os.Stat(blobPath); err == nil && !bInfo.IsDir() {
					return blobPath, nil
				}
			}
		}
	}

	return modelPathOrTag, fmt.Errorf("model file not found: '%s' (neither as direct path nor as installed Ollama model)", modelPathOrTag)
}
