package sampler

import (
	"math/rand"
	"testing"
)

func TestSampleGreedy(t *testing.T) {
	logits := []float32{1.0, 5.0, 2.0, 0.5}
	params := Params{Temperature: 0.0}

	selected := SampleToken(logits, nil, params)
	if selected != 1 {
		t.Errorf("Greedy sampling failed: expected token 1, got %d", selected)
	}
}

func TestSampleRepetitionPenalty(t *testing.T) {
	// Logits with token 0 having higher logit, but token 0 is in history with high penalty
	logits := []float32{4.0, 3.8, 1.0}
	history := []int{0}
	params := Params{
		Temperature: 0.0,
		RepPenalty:  2.0, // 4.0 / 2.0 = 2.0 < 3.8
	}

	selected := SampleToken(logits, history, params)
	if selected != 1 {
		t.Errorf("Repetition penalty failed: expected token 1, got %d", selected)
	}
}

func TestSampleTopP(t *testing.T) {
	logits := []float32{10.0, 1.0, 0.0}
	params := Params{
		Temperature: 1.0,
		TopP:        0.9,
		Rand:        rand.New(rand.NewSource(42)),
	}

	selected := SampleToken(logits, nil, params)
	if selected != 0 {
		t.Errorf("TopP sampling failed: expected token 0, got %d", selected)
	}
}

func TestSamplerBoundaryConditions(t *testing.T) {
	// Empty logits
	if SampleToken(nil, nil, Params{}) != 0 {
		t.Errorf("Expected 0 for empty logits")
	}

	// Single logit
	if SampleToken([]float32{42.0}, nil, Params{}) != 0 {
		t.Errorf("Expected 0 for single element logit")
	}

	// TopK = 1
	logits := []float32{1.0, 5.0, 2.0}
	selected := SampleToken(logits, nil, Params{Temperature: 1.0, TopK: 1})
	if selected != 1 {
		t.Errorf("TopK=1 failed: expected token 1, got %d", selected)
	}

	// TopK > len(logits)
	selected = SampleToken(logits, nil, Params{Temperature: 1.0, TopK: 100})
	if selected < 0 || selected >= len(logits) {
		t.Errorf("TopK > len failed: got out of range token %d", selected)
	}
}

func TestSamplerRepetitionPenaltyEdgeCases(t *testing.T) {
	// Negative logit repetition penalty multiplication
	logits := []float32{-2.0, -5.0}
	history := []int{0}
	params := Params{
		Temperature: 0.0,
		RepPenalty:  2.0, // -2.0 * 2.0 = -4.0 > -5.0
	}
	selected := SampleToken(logits, history, params)
	if selected != 0 {
		t.Errorf("Negative rep penalty failed: expected token 0, got %d", selected)
	}

	// History token with index out of bounds
	historyOOB := []int{-1, 9999}
	selected = SampleToken(logits, historyOOB, params)
	if selected != 0 {
		t.Errorf("OOB history failed: expected token 0, got %d", selected)
	}
}

func BenchmarkSamplerTopK(b *testing.B) {
	logits := make([]float32, 152064)
	for i := range logits {
		logits[i] = float32(i % 100)
	}
	logits[42] = 500.0

	params := Params{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SampleToken(logits, nil, params)
	}
}
