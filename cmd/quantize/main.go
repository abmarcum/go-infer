package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"go-inference/pkg/gguf"
	"go-inference/pkg/quant"
	"log"
	"os"
	"strings"
	"time"
)

func main() {
	var (
		inputPath  string
		outputPath string
		targetType string
	)

	flag.StringVar(&inputPath, "input", "", "Path to input GGUF model file")
	flag.StringVar(&outputPath, "output", "", "Path to output quantized GGUF model file")
	flag.StringVar(&targetType, "type", "q4_0", "Target quantization format: q4_0, q8_0")
	flag.Parse()

	if inputPath == "" || outputPath == "" {
		fmt.Fprintf(os.Stderr, "GGUF Quantization Tool - Pure Go\n\n")
		fmt.Fprintf(os.Stderr, "Usage: quantize -input <model.gguf> -output <model-q4_0.gguf> [-type q4_0|q8_0]\n\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	start := time.Now()
	log.Printf("Opening input model: %s", inputPath)
	reader, err := gguf.OpenFile(inputPath)
	if err != nil {
		log.Fatalf("Failed to open GGUF file: %v", err)
	}
	defer reader.Close()

	log.Printf("Model contains %d tensors. Target quantization: %s", len(reader.Header.Tensors), targetType)

	var targetGGMLType uint32
	switch strings.ToLower(targetType) {
	case "q8_0":
		targetGGMLType = gguf.GGMLTypeQ8_0
	case "q4_0":
		targetGGMLType = gguf.GGMLTypeQ4_0
	default:
		log.Fatalf("Unsupported target type: %s (choose q4_0 or q8_0)", targetType)
	}

	// Prepare output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer outFile.Close()

	// Write GGUF Magic and Version
	binary.Write(outFile, binary.LittleEndian, uint32(gguf.Magic))
	binary.Write(outFile, binary.LittleEndian, uint32(gguf.Version3))
	binary.Write(outFile, binary.LittleEndian, uint64(len(reader.Header.Tensors)))
	binary.Write(outFile, binary.LittleEndian, uint64(0)) // 0 metadata entries for compact export

	// Calculate tensor data offsets and write tensor headers
	var tensorHeadersBuf bytes.Buffer
	currentDataOffset := uint64(0)

	type TensorEntry struct {
		Name     string
		Dims     []uint64
		OrigType uint32
		NewType  uint32
		Data     []byte
		Offset   uint64
	}

	var entries []TensorEntry
	for name, info := range reader.Header.Tensors {
		rawData, _, err := reader.GetTensorData(name)
		if err != nil {
			log.Fatalf("Failed to read tensor %s: %v", name, err)
		}

		newType := info.Type
		var processedData []byte

		// Only quantize 2D weight matrices (e.g. blk.*.attn_*, blk.*.ffn_*, output.weight)
		shouldQuantize := len(info.Dimensions) >= 2 && !strings.Contains(name, "norm")
		if shouldQuantize && (info.Type == gguf.GGMLTypeF32 || info.Type == gguf.GGMLTypeF16) {
			newType = targetGGMLType
			var f32s []float32
			if info.Type == gguf.GGMLTypeF32 {
				f32s = quant.DequantizeF32(rawData, info.NumElements())
			} else {
				f32s = quant.DequantizeF16(rawData, info.NumElements())
			}

			if targetGGMLType == gguf.GGMLTypeQ4_0 {
				processedData = quant.QuantizeQ4_0(f32s)
			} else {
				processedData = quant.QuantizeQ8_0(f32s)
			}
		} else {
			processedData = rawData
		}

		// Align offset to 32 bytes
		alignedOffset := (currentDataOffset + 31) &^ 31
		padding := alignedOffset - currentDataOffset
		if padding > 0 {
			processedData = append(make([]byte, padding), processedData...)
		}

		entries = append(entries, TensorEntry{
			Name:     name,
			Dims:     info.Dimensions,
			OrigType: info.Type,
			NewType:  newType,
			Data:     processedData,
			Offset:   alignedOffset,
		})

		currentDataOffset = alignedOffset + uint64(len(processedData))

		// Write tensor info header
		binary.Write(&tensorHeadersBuf, binary.LittleEndian, uint64(len(name)))
		tensorHeadersBuf.WriteString(name)
		binary.Write(&tensorHeadersBuf, binary.LittleEndian, uint32(len(info.Dimensions)))
		for _, d := range info.Dimensions {
			binary.Write(&tensorHeadersBuf, binary.LittleEndian, d)
		}
		binary.Write(&tensorHeadersBuf, binary.LittleEndian, newType)
		binary.Write(&tensorHeadersBuf, binary.LittleEndian, alignedOffset)
	}

	// Write headers to output
	outFile.Write(tensorHeadersBuf.Bytes())

	// Pad to 32-byte alignment before tensor data
	curPos, _ := outFile.Seek(0, os.SEEK_CUR)
	dataStart := (curPos + 31) &^ 31
	padLen := dataStart - curPos
	if padLen > 0 {
		outFile.Write(make([]byte, padLen))
	}

	// Write all tensor data
	for _, entry := range entries {
		outFile.Write(entry.Data)
	}

	totalDur := time.Since(start)
	log.Printf("Successfully quantized model to %s in %.2fs!", outputPath, totalDur.Seconds())
}
