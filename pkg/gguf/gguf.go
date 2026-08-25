package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Reader reads and navigates GGUF files.
type Reader struct {
	Header   Header
	Data     []byte
	filePath string
	file     *os.File
}

// OpenFile opens and parses a GGUF file using mmap.
func OpenFile(filePath string) (*Reader, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open GGUF file: %w", err)
	}

	data, err := MmapFile(file)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap GGUF file: %w", err)
	}

	r := bytes.NewReader(data)
	hdr, err := parseHeader(r)
	if err != nil {
		MunmapFile(data)
		file.Close()
		return nil, fmt.Errorf("failed to parse GGUF header: %w", err)
	}

	return &Reader{
		Header:   *hdr,
		Data:     data,
		filePath: filePath,
		file:     file,
	}, nil
}

// Close unmaps the memory and closes the file handle.
func (r *Reader) Close() error {
	var err1, err2 error
	if r.Data != nil {
		err1 = MunmapFile(r.Data)
		r.Data = nil
	}
	if r.file != nil {
		err2 = r.file.Close()
		r.file = nil
	}
	if err1 != nil {
		return err1
	}
	return err2
}

// GetTensorData returns the raw byte slice for a given tensor.
func (r *Reader) GetTensorData(tensorName string) ([]byte, *TensorInfo, error) {
	info, ok := r.Header.Tensors[tensorName]
	if !ok {
		return nil, nil, fmt.Errorf("tensor not found: %s", tensorName)
	}

	startOffset := r.Header.DataOffset + int64(info.Offset)
	if startOffset < 0 || startOffset >= int64(len(r.Data)) {
		return nil, nil, fmt.Errorf("tensor %s offset %d out of bounds (file size %d)", tensorName, startOffset, len(r.Data))
	}

	byteSize := TensorByteSize(info.Type, info.NumElements())
	endOffset := startOffset + int64(byteSize)
	if endOffset > int64(len(r.Data)) {
		return nil, nil, fmt.Errorf("tensor %s range [%d, %d] exceeds file size %d", tensorName, startOffset, endOffset, len(r.Data))
	}

	return r.Data[startOffset:endOffset], &info, nil
}

// TensorByteSize calculates the byte size of a tensor given its type and total number of elements.
func TensorByteSize(tType uint32, numElements int) int {
	switch tType {
	case GGMLTypeF32:
		return numElements * 4
	case GGMLTypeF16, GGMLTypeBF16:
		return numElements * 2
	case GGMLTypeQ4_0:
		return ((numElements + 31) / 32) * 18
	case GGMLTypeQ4_1:
		return ((numElements + 31) / 32) * 20
	case GGMLTypeQ5_0:
		return ((numElements + 31) / 32) * 22
	case GGMLTypeQ5_1:
		return ((numElements + 31) / 32) * 24
	case GGMLTypeQ8_0:
		return ((numElements + 31) / 32) * 34
	case GGMLTypeQ8_1:
		return ((numElements + 31) / 32) * 40
	case GGMLTypeQ2_K:
		return ((numElements + 255) / 256) * 84
	case GGMLTypeQ3_K:
		return ((numElements + 255) / 256) * 110
	case GGMLTypeQ4_K:
		return ((numElements + 255) / 256) * 144
	case GGMLTypeQ5_K:
		return ((numElements + 255) / 256) * 176
	case GGMLTypeQ6_K:
		return ((numElements + 255) / 256) * 210
	case GGMLTypeQ8_K:
		return ((numElements + 255) / 256) * 292
	case GGMLTypeIQ2_XXS:
		return ((numElements + 255) / 256) * 66
	case GGMLTypeIQ2_XS:
		return ((numElements + 255) / 256) * 74
	case GGMLTypeIQ3_XXS:
		return ((numElements + 255) / 256) * 98
	case GGMLTypeIQ1_S:
		return ((numElements + 255) / 256) * 48
	case GGMLTypeIQ4_NL:
		return ((numElements + 31) / 32) * 18
	case GGMLTypeIQ3_S:
		return ((numElements + 255) / 256) * 110
	case GGMLTypeIQ2_S:
		return ((numElements + 255) / 256) * 82
	case GGMLTypeIQ4_XS:
		return ((numElements + 255) / 256) * 136
	case GGMLTypeI8:
		return numElements
	case GGMLTypeI16:
		return numElements * 2
	case GGMLTypeI32:
		return numElements * 4
	case GGMLTypeI64, GGMLTypeF64:
		return numElements * 8
	default:
		return numElements * 4
	}
}

func parseHeader(r *bytes.Reader) (*Header, error) {
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != Magic {
		return nil, fmt.Errorf("invalid magic: 0x%08X (expected 0x%08X)", magic, Magic)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version != Version2 && version != Version3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	var tensorCount, kvCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
		return nil, fmt.Errorf("read kv count: %w", err)
	}

	metadata := make(map[string]interface{}, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(r)
		if err != nil {
			return nil, fmt.Errorf("read metadata key %d: %w", i, err)
		}
		var valType uint32
		if err := binary.Read(r, binary.LittleEndian, &valType); err != nil {
			return nil, fmt.Errorf("read metadata valType for %s: %w", key, err)
		}
		val, err := readMetadataValue(r, valType)
		if err != nil {
			return nil, fmt.Errorf("read metadata value for %s: %w", key, err)
		}
		metadata[key] = val
	}

	tensors := make(map[string]TensorInfo, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readString(r)
		if err != nil {
			return nil, fmt.Errorf("read tensor name %d: %w", i, err)
		}
		var nDims uint32
		if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
			return nil, fmt.Errorf("read tensor %s nDims: %w", name, err)
		}
		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			if err := binary.Read(r, binary.LittleEndian, &dims[d]); err != nil {
				return nil, fmt.Errorf("read tensor %s dim %d: %w", name, d, err)
			}
		}
		var tType uint32
		var offset uint64
		if err := binary.Read(r, binary.LittleEndian, &tType); err != nil {
			return nil, fmt.Errorf("read tensor %s type: %w", name, err)
		}
		if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("read tensor %s offset: %w", name, err)
		}

		tensors[name] = TensorInfo{
			Name:       name,
			Dimensions: dims,
			Type:       tType,
			Offset:     offset,
		}
	}

	curPos, _ := r.Seek(0, io.SeekCurrent)
	alignment := int64(32)
	if alg, ok := metadata["general.alignment"].(uint32); ok && alg > 0 {
		alignment = int64(alg)
	}

	padding := (alignment - (curPos % alignment)) % alignment
	dataOffset := curPos + padding

	return &Header{
		Version:     version,
		TensorCount: tensorCount,
		KVCount:     kvCount,
		Metadata:    metadata,
		Tensors:     tensors,
		DataOffset:  dataOffset,
	}, nil
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readMetadataValue(r io.Reader, valType uint32) (interface{}, error) {
	switch valType {
	case TypeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeFloat32:
		var v float32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return false, err
		}
		return v != 0, nil
	case TypeString:
		return readString(r)
	case TypeArray:
		var elemType uint32
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		arr := make([]interface{}, count)
		for i := uint64(0); i < count; i++ {
			elem, err := readMetadataValue(r, elemType)
			if err != nil {
				return nil, err
			}
			arr[i] = elem
		}
		return arr, nil
	case TypeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeFloat64:
		var v float64
		return v, binary.Read(r, binary.LittleEndian, &v)
	default:
		return nil, fmt.Errorf("unsupported metadata type: %d", valType)
	}
}

// Metadata helper functions
func GetMetadataUint(m map[string]interface{}, key string, def uint64) uint64 {
	val, ok := m[key]
	if !ok {
		return def
	}
	switch v := val.(type) {
	case uint8:
		return uint64(v)
	case uint16:
		return uint64(v)
	case uint32:
		return uint64(v)
	case uint64:
		return v
	case int8:
		return uint64(v)
	case int16:
		return uint64(v)
	case int32:
		return uint64(v)
	case int64:
		return uint64(v)
	default:
		return def
	}
}

func GetMetadataFloat(m map[string]interface{}, key string, def float64) float64 {
	val, ok := m[key]
	if !ok {
		return def
	}
	switch v := val.(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	default:
		return def
	}
}

func GetMetadataString(m map[string]interface{}, key string, def string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return def
}

func GetMetadataStringArray(m map[string]interface{}, key string) []string {
	raw, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	res := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok {
			res = append(res, s)
		}
	}
	return res
}
