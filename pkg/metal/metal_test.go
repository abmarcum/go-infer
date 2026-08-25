package metal

import (
	"bytes"
	"encoding/binary"
	"go-inference/pkg/quant"
	"math"
	"testing"
)

func TestMetalInitialization(t *testing.T) {
	err := Init()
	if err != nil {
		t.Skipf("Skipping Metal test on unsupported system: %v", err)
	}
	if !IsAvailable() {
		t.Fatalf("Expected Metal to be available after Init()")
	}
}

func TestMetalGEMV_F32(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip("Metal not available")
	}

	rows := 4
	cols := 32
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i + 1)
	}

	w := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if r == c {
				w[r*cols+c] = 1.0
			}
		}
	}

	y := make([]float32, rows)
	if err := MatMulF32(y, x, w, rows, cols); err != nil {
		t.Fatalf("Metal F32 GEMV failed: %v", err)
	}

	for r := 0; r < rows; r++ {
		if math.Abs(float64(y[r]-x[r])) > 1e-4 {
			t.Errorf("Mismatch at row %d: got %f, expected %f", r, y[r], x[r])
		}
	}
}

func TestMetalGEMV_Q4_0(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip("Metal not available")
	}

	rows := 2
	cols := 32
	x := make([]float32, cols)
	for i := range x {
		x[i] = 1.0
	}

	var buf bytes.Buffer
	for r := 0; r < rows; r++ {
		scale := quant.FP32ToFP16(0.5)
		binary.Write(&buf, binary.LittleEndian, scale)
		var qs [16]byte
		for j := 0; j < 16; j++ {
			qs[j] = byte(10 | (10 << 4)) // (10-8)*0.5 = 1.0 per element
		}
		buf.Write(qs[:])
	}
	rawQ4 := buf.Bytes()

	y := make([]float32, rows)
	if err := MatMulQ4_0(y, x, rawQ4, rows, cols); err != nil {
		t.Fatalf("Metal Q4_0 GEMV failed: %v", err)
	}

	for r := 0; r < rows; r++ {
		// Expected sum = 32 * 1.0 * 1.0 = 32.0
		if math.Abs(float64(y[r]-32.0)) > 1e-3 {
			t.Errorf("Q4_0 mismatch at row %d: got %f, expected 32.0", r, y[r])
		}
	}
}

func TestMetalGEMV_Q8_0(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip("Metal not available")
	}

	rows := 4
	cols := 32
	x := make([]float32, cols)
	for i := range x {
		x[i] = 1.0
	}

	var buf bytes.Buffer
	for r := 0; r < rows; r++ {
		scale := quant.FP32ToFP16(0.25)
		binary.Write(&buf, binary.LittleEndian, scale)
		var qs [32]int8
		for j := 0; j < 32; j++ {
			qs[j] = 4 // 4 * 0.25 = 1.0
		}
		binary.Write(&buf, binary.LittleEndian, qs)
	}
	rawQ8 := buf.Bytes()

	y := make([]float32, rows)
	if err := MatMulQ8_0(y, x, rawQ8, rows, cols); err != nil {
		t.Fatalf("Metal Q8_0 GEMV failed: %v", err)
	}

	for r := 0; r < rows; r++ {
		if math.Abs(float64(y[r]-32.0)) > 1e-3 {
			t.Errorf("Q8_0 mismatch at row %d: got %f, expected 32.0", r, y[r])
		}
	}
}

func TestMetalGEMV_Q4_K(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip("Metal not available")
	}

	rows := 2
	cols := 256
	x := make([]float32, cols)
	for i := range x {
		x[i] = 1.0
	}

	rawQ4K := make([]byte, rows*144)
	for r := 0; r < rows; r++ {
		offset := r * 144
		binary.LittleEndian.PutUint16(rawQ4K[offset:offset+2], quant.FP32ToFP16(1.0))
		binary.LittleEndian.PutUint16(rawQ4K[offset+2:offset+4], quant.FP32ToFP16(0.0))
		// Set scales to 1
		for s := 0; s < 12; s++ {
			rawQ4K[offset+4+s] = 1
		}
		// Set qs to 2 (2 - 0 = 2)
		for q := 0; q < 128; q++ {
			rawQ4K[offset+16+q] = 0x22
		}
	}

	y := make([]float32, rows)
	if err := MatMulQ4_K(y, x, rawQ4K, rows, cols); err != nil {
		t.Fatalf("Metal Q4_K GEMV failed: %v", err)
	}

	for r := 0; r < rows; r++ {
		if math.Abs(float64(y[r]-512.0)) > 1e-2 {
			t.Errorf("Q4_K mismatch at row %d: got %f, expected 512.0", r, y[r])
		}
	}
}

func TestMetalRMSNorm(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip("Metal not available")
	}

	dim := 64
	x := make([]float32, dim)
	weight := make([]float32, dim)
	for i := 0; i < dim; i++ {
		x[i] = 2.0
		weight[i] = 1.0
	}

	out := make([]float32, dim)
	if err := RMSNorm(out, x, weight, dim, 1e-5); err != nil {
		t.Fatalf("Metal RMSNorm failed: %v", err)
	}

	for i := 0; i < dim; i++ {
		if math.Abs(float64(out[i]-1.0)) > 1e-3 {
			t.Errorf("RMSNorm mismatch at %d: got %f, expected 1.0", i, out[i])
		}
	}
}

func BenchmarkMetalGEMV_Q4_K(b *testing.B) {
	if err := Init(); err != nil {
		b.Skip("Metal not available")
	}

	rows := 4096
	cols := 4096
	x := make([]float32, cols)
	raw := make([]byte, rows*(cols/256)*144)

	y := make([]float32, rows)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulQ4_K(y, x, raw, rows, cols)
	}
}
