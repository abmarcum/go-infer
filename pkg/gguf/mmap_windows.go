//go:build windows

package gguf

import (
	"io"
	"os"
)

// MmapFile on Windows falls back to reading the entire file into memory.
func MmapFile(file *os.File) ([]byte, error) {
	return io.ReadAll(file)
}

// MunmapFile is a no-op on Windows fallback.
func MunmapFile(data []byte) error {
	return nil
}
