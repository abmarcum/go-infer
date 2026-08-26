package downloader

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// HFModelInfo contains Hugging Face Model API metadata.
type HFModelInfo struct {
	Siblings []struct {
		RFilename string `json:"rfilename"`
	} `json:"siblings"`
}

// ProgressFunc reports download metrics.
type ProgressFunc func(downloaded, total int64, percent float64, speedMBps float64)

// DownloadHuggingFaceGGUF downloads a GGUF model directly from Hugging Face Hub.
func DownloadHuggingFaceGGUF(target string, destDir string, onProgress ProgressFunc) (string, error) {
	if destDir == "" {
		destDir = "models"
	}
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return "", fmt.Errorf("create dest dir: %w", err)
	}

	repo, filename, err := resolveHFRepoAndFile(target)
	if err != nil {
		return "", err
	}

	// Security: Sanitize filename to prevent directory traversal
	filename = filepath.Base(filename)
	if filename == "." || filename == "/" || filename == "" {
		return "", fmt.Errorf("invalid filename resolved: %s", filename)
	}

	downloadURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s?download=true", repo, filename)
	destPath := filepath.Join(destDir, filename)

	// Check if already completely downloaded
	if fi, err := os.Stat(destPath); err == nil && fi.Size() > 1024*1024 {
		return destPath, nil
	}

	client := &http.Client{Timeout: 0} // unlimited timeout for large models
	req, err := http.NewRequest("GET", downloadURL, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("User-Agent", "go-infer/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("connect to Hugging Face: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("hugging face returned HTTP %s for URL: %s", resp.Status, downloadURL)
	}

	totalSize := resp.ContentLength
	out, err := os.Create(destPath + ".part")
	if err != nil {
		return "", fmt.Errorf("create part file: %w", err)
	}
	defer out.Close()

	buf := make([]byte, 1024*1024) // 1MB buffer
	var downloaded int64
	startTime := time.Now()
	lastReport := time.Now()

	for {
		nr, er := resp.Body.Read(buf)
		if nr > 0 {
			nw, ew := out.Write(buf[0:nr])
			if ew != nil {
				return "", fmt.Errorf("write file: %w", ew)
			}
			if nr != nw {
				return "", io.ErrShortWrite
			}
			downloaded += int64(nw)

			if onProgress != nil && time.Since(lastReport) > 200*time.Millisecond {
				elapsed := time.Since(startTime).Seconds()
				speed := 0.0
				if elapsed > 0 {
					speed = (float64(downloaded) / (1024 * 1024)) / elapsed
				}
				percent := 0.0
				if totalSize > 0 {
					percent = (float64(downloaded) / float64(totalSize)) * 100.0
				}
				onProgress(downloaded, totalSize, percent, speed)
				lastReport = time.Now()
			}
		}
		if er != nil {
			if er != io.EOF {
				return "", fmt.Errorf("stream read error: %w", er)
			}
			break
		}
	}

	out.Close()
	if err := os.Rename(destPath+".part", destPath); err != nil {
		return "", fmt.Errorf("finalize downloaded file: %w", err)
	}

	return destPath, nil
}

// resolveHFRepoAndFile parses user string (repo or repo/file) and discovers suitable GGUF file.
func resolveHFRepoAndFile(target string) (repo string, filename string, err error) {
	target = strings.TrimPrefix(target, "https://huggingface.co/")
	target = strings.Trim(target, "/")

	parts := strings.Split(target, "/")
	if len(parts) < 2 {
		return "", "", fmt.Errorf("invalid Hugging Face repository format (expected 'owner/repo' or 'owner/repo/model.gguf')")
	}

	repo = parts[0] + "/" + parts[1]
	if len(parts) >= 3 && strings.HasSuffix(parts[len(parts)-1], ".gguf") {
		filename = parts[len(parts)-1]
		return repo, filename, nil
	}

	// Query Hugging Face Model API to find GGUF files in repo
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s", repo)
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(apiURL)
	if err != nil {
		return "", "", fmt.Errorf("query repo metadata (%s): %w", repo, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("hugging face repo not found (%s)", repo)
	}

	var info HFModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", "", fmt.Errorf("decode model info: %w", err)
	}

	var ggufFiles []string
	for _, s := range info.Siblings {
		if strings.HasSuffix(strings.ToLower(s.RFilename), ".gguf") {
			ggufFiles = append(ggufFiles, s.RFilename)
		}
	}

	if len(ggufFiles) == 0 {
		return "", "", fmt.Errorf("no .gguf files found in repository %s", repo)
	}

	// Prefer Q4_K_M or Q4_0 or Q8_0 or first available
	for _, pref := range []string{"q4_k_m", "q4_k", "q4_0", "q8_0"} {
		for _, f := range ggufFiles {
			if strings.Contains(strings.ToLower(f), pref) {
				return repo, f, nil
			}
		}
	}

	return repo, ggufFiles[0], nil
}
