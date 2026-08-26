package sampler

import (
	"testing"
)

func TestJSONGrammarValidation(t *testing.T) {
	v := NewJSONGrammarValidator()

	validSequence := []string{"{", "\"", "name", "\"", ":", " ", "\"", "Alice", "\"", ",", "\"", "age", "\"", ":", " ", "30", "}"}
	for _, tok := range validSequence {
		if !v.Accepts(tok) {
			t.Fatalf("Grammar validator incorrectly rejected token %q in state %v", tok, v.State)
		}
		v.Process(tok)
	}

	if v.State != JSONStateDone {
		t.Errorf("Expected JSONStateDone, got %v", v.State)
	}
}

func TestJSONGrammarRejection(t *testing.T) {
	v := NewJSONGrammarValidator()
	v.Process("{\"key\":")

	// Must reject trailing comma before value or invalid raw tokens
	if v.Accepts(",") {
		t.Errorf("Grammar validator should reject comma right after colon")
	}

	// Should accept string value
	if !v.Accepts("\"value\"") {
		t.Errorf("Grammar validator should accept valid string value")
	}
}

func TestApplyJSONGrammarMask(t *testing.T) {
	v := NewJSONGrammarValidator()
	vocab := []string{"{", "hello", "}", "\"", ":"}
	logits := []float32{1.0, 5.0, 2.0, 3.0, 4.0}

	// At start, only '{' is valid
	ApplyJSONGrammarMask(logits, vocab, v)

	if logits[0] <= -1e8 {
		t.Errorf("Expected '{' (index 0) to be allowed, got logit %f", logits[0])
	}
	if logits[1] > -1e8 {
		t.Errorf("Expected 'hello' (index 1) to be masked out at start of JSON, got logit %f", logits[1])
	}
}
