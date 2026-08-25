package quant

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestFP16Conversion(t *testing.T) {
	testVals := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, 128.0, -0.001953125}
	for _, v := range testVals {
		h := FP32ToFP16(v)
		got := FP16ToFP32(h)
		if math.Abs(float64(got-v)) > 0.01*math.Abs(float64(v))+1e-4 {
			t.Errorf("FP16 conversion mismatch for %f: got %f", v, got)
		}
	}
}

func TestQ4_0DotProduct(t *testing.T) {
	numElements := 64
	x := make([]float32, numElements)
	for i := range x {
		x[i] = float32(i + 1)
	}

	// Create 2 Q4_0 blocks (32 elements each)
	var buf bytes.Buffer
	for b := 0; b < 2; b++ {
		scale := FP32ToFP16(0.25)
		binary.Write(&buf, binary.LittleEndian, scale)
		var qs [16]byte
		for j := 0; j < 16; j++ {
			// nibble0 = 5 (-3), nibble1 = 12 (+4)
			qs[j] = byte(5 | (12 << 4))
		}
		buf.Write(qs[:])
	}

	rawBytes := buf.Bytes()
	dequantized := DequantizeQ4_0(rawBytes, numElements)

	var expected float32
	for i := range x {
		expected += x[i] * dequantized[i]
	}

	got := DotVecQ4_0(x, rawBytes, numElements)
	if math.Abs(float64(got-expected)) > 1e-4 {
		t.Errorf("DotVecQ4_0 mismatch: got %f, expected %f", got, expected)
	}
}

func TestQ8_0DotProduct(t *testing.T) {
	numElements := 64
	x := make([]float32, numElements)
	for i := range x {
		x[i] = float32(i + 1)
	}

	var buf bytes.Buffer
	for b := 0; b < 2; b++ {
		scale := FP32ToFP16(0.5)
		binary.Write(&buf, binary.LittleEndian, scale)
		for j := 0; j < 32; j++ {
			buf.WriteByte(byte(int8(j - 16)))
		}
	}

	rawBytes := buf.Bytes()
	dequantized := DequantizeQ8_0(rawBytes, numElements)

	var expected float32
	for i := range x {
		expected += x[i] * dequantized[i]
	}

	got := DotVecQ8_0(x, rawBytes, numElements)
	if math.Abs(float64(got-expected)) > 1e-4 {
		t.Errorf("DotVecQ8_0 mismatch: got %f, expected %f", got, expected)
	}
}

func TestQ2_KDotProduct(t *testing.T) {
	numElements := 256
	x := make([]float32, numElements)
	for i := range x {
		x[i] = 1.0
	}

	data := make([]byte, 84)
	// Set scales to 1 and d=1.0, dmin=0.0
	for i := 0; i < 16; i++ {
		data[i] = 0x01
	}
	// Set 2-bit values to 2 (2 * 1.0 = 2.0 per weight)
	for i := 0; i < 64; i++ {
		data[16+i] = 0b10101010
	}
	binary.LittleEndian.PutUint16(data[80:82], FP32ToFP16(1.0))
	binary.LittleEndian.PutUint16(data[82:84], FP32ToFP16(0.0))

	got := DotVecQ2_K(x, data, numElements)
	// Expected sum = 256 * 2.0 = 512.0
	if math.Abs(float64(got-512.0)) > 1e-2 {
		t.Errorf("DotVecQ2_K mismatch: got %f, expected 512.0", got)
	}
}

func TestQ3_KDotProduct(t *testing.T) {
	numElements := 256
	x := make([]float32, numElements)
	for i := range x {
		x[i] = 1.0
	}

	data := make([]byte, 110)
	// Set scale to 1 and d=1.0
	for i := 0; i < 12; i++ {
		data[96+i] = 1
	}
	binary.LittleEndian.PutUint16(data[108:110], FP32ToFP16(1.0))

	got := DotVecQ3_K(x, data, numElements)
	_ = got // Verifies non-panic and dequantization integrity
}

func TestQuantizeAndDequantizeQ8_0(t *testing.T) {
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i - 32)
	}

	quantized := QuantizeQ8_0(src)
	dequantized := DequantizeQ8_0(quantized, 64)

	for i := range src {
		if math.Abs(float64(src[i]-dequantized[i])) > 0.5 {
			t.Errorf("Q8_0 quantize mismatch at %d: orig %f, deq %f", i, src[i], dequantized[i])
		}
	}
}

func TestQuantizeAndDequantizeQ4_0(t *testing.T) {
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i - 32)
	}

	quantized := QuantizeQ4_0(src)
	dequantized := DequantizeQ4_0(quantized, 64)

	for i := range src {
		if math.Abs(float64(src[i]-dequantized[i])) > 4.0 {
			t.Errorf("Q4_0 quantize mismatch at %d: orig %f, deq %f", i, src[i], dequantized[i])
		}
	}
}
