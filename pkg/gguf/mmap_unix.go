//go:build !windows

package gguf

import (
	"fmt"
	"os"
	"syscall"
)

// MmapFile memory-maps a file read-only on Unix-like systems.
func MmapFile(file *os.File) ([]byte, error) {
	stat, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat failed: %w", err)
	}

	size := stat.Size()
	if size == 0 {
		return []byte{}, nil
	}

	data, err := syscall.Mmap(int(file.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap failed: %w", err)
	}

	return data, nil
}

// MunmapFile unmaps the mapped byte slice.
func MunmapFile(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return syscall.Munmap(data)
}
