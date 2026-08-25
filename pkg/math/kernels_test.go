package math

import (
	"math"
	"testing"
)

func TestRMSNorm(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0, 4.0}
	w := []float32{1.0, 1.0, 1.0, 1.0}
	out := make([]float32, len(x))

	RMSNorm(out, x, w, 1e-5)

	// Mean(x^2) = (1 + 4 + 9 + 16)/4 = 30/4 = 7.5
	// scale = 1 / sqrt(7.5) = ~0.36514837
	expected0 := float32(1.0 / math.Sqrt(7.5+1e-5))
	if math.Abs(float64(out[0]-expected0)) > 1e-4 {
		t.Errorf("RMSNorm out[0] mismatch: got %f, expected %f", out[0], expected0)
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0}
	Softmax(x)

	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum is not 1.0: got %f", sum)
	}
	if x[2] <= x[1] || x[1] <= x[0] {
		t.Errorf("Softmax ordering violated: %v", x)
	}
}

func TestApplyRoPE(t *testing.T) {
	vec := []float32{1.0, 0.0, 0.0, 1.0}
	ApplyRoPE(vec, 0, 4, 10000.0)
	// At pos 0, theta^0 = 1, cos(0) = 1, sin(0) = 0 -> vec should be unchanged
	if vec[0] != 1.0 || vec[1] != 0.0 || vec[2] != 0.0 || vec[3] != 1.0 {
		t.Errorf("RoPE at pos 0 changed values: %v", vec)
	}
}

func TestSwiGLU(t *testing.T) {
	gate := []float32{0.0, 2.0}
	up := []float32{3.0, 4.0}
	SwiGLU(gate, up, 2)
	// silu(0) = 0 -> gate[0] = 0
	if gate[0] != 0 {
		t.Errorf("SwiGLU gate[0] expected 0, got %f", gate[0])
	}
	// silu(2) = 2 / (1 + exp(-2)) = 2 / 1.135335 = ~1.76159
	// gate[1] = 1.76159 * 4 = ~7.04637
	if gate[1] <= 6.0 || gate[1] >= 8.0 {
		t.Errorf("SwiGLU gate[1] unexpected: %f", gate[1])
	}
}

func TestGEMVMatMulF32(t *testing.T) {
	engine := NewGEMVEngine(4)
	rows := 4
	cols := 4
	x := []float32{1, 2, 3, 4}
	// Identity matrix
	w := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	y := make([]float32, rows)
	engine.MatMulF32(y, x, w, rows, cols)

	for i := 0; i < rows; i++ {
		if y[i] != x[i] {
			t.Errorf("GEMV identity mismatch at index %d: got %f, expected %f", i, y[i], x[i])
		}
	}
}
