package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestGGUFCorruptedHeader(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. Truncated File
	truncPath := filepath.Join(tmpDir, "trunc.gguf")
	_ = os.WriteFile(truncPath, []byte{0x47, 0x47, 0x55, 0x46}, 0644)
	_, err := OpenFile(truncPath)
	if err == nil {
		t.Errorf("Expected error opening truncated GGUF file")
	}

	// 2. Invalid Magic Bytes
	badMagicPath := filepath.Join(tmpDir, "badmagic.gguf")
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(0x12345678)) // bad magic
	binary.Write(&buf, binary.LittleEndian, uint32(Version3))
	binary.Write(&buf, binary.LittleEndian, uint64(0))
	binary.Write(&buf, binary.LittleEndian, uint64(0))
	_ = os.WriteFile(badMagicPath, buf.Bytes(), 0644)

	_, err = OpenFile(badMagicPath)
	if err == nil {
		t.Errorf("Expected error opening GGUF file with bad magic bytes")
	}
}

func TestGGUFMetadataHelpers(t *testing.T) {
	meta := map[string]interface{}{
		"uint_val":   uint64(42),
		"float_val":  float64(3.1415),
		"string_val": "test_string",
	}

	if GetMetadataUint(meta, "uint_val", 0) != 42 {
		t.Errorf("Expected 42 from GetMetadataUint")
	}
	if GetMetadataUint(meta, "missing_val", 99) != 99 {
		t.Errorf("Expected fallback 99 from GetMetadataUint")
	}

	if GetMetadataFloat(meta, "float_val", 0.0) != 3.1415 {
		t.Errorf("Expected 3.1415 from GetMetadataFloat")
	}
	if GetMetadataFloat(meta, "missing_val", 1.0) != 1.0 {
		t.Errorf("Expected fallback 1.0 from GetMetadataFloat")
	}

	if GetMetadataString(meta, "string_val", "") != "test_string" {
		t.Errorf("Expected 'test_string' from GetMetadataString")
	}
}
