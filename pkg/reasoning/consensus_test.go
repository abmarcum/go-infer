package reasoning

import (
	"testing"
)

func TestExtractAnswer(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{
			input: "Therefore, the final result is \\boxed{42}.",
			want:  "42",
		},
		{
			input: "Hence, the solution set is \\boxed{\\frac{3}{4}}.",
			want:  "\\frac{3}{4}",
		},
		{
			input: "Step 1: 5 * 10 = 50\n#### 50",
			want:  "50",
		},
		{
			input: "Let us calculate the total.\nThe answer is: 128",
			want:  "128",
		},
		{
			input: "Final Answer: 3.14159",
			want:  "3.14159",
		},
	}

	for _, tt := range tests {
		got := ExtractAnswer(tt.input)
		if got != tt.want {
			t.Errorf("ExtractAnswer(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestNormalizeAnswer(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"42", "42"},
		{"42.0", "42"},
		{"$1,250", "1250"},
		{"1/2", "0.5"},
		{"2/4", "0.5"},
		{"\\frac{1}{2}", "0.5"},
		{"\\boxed{100}", "100"},
		{" 42.00 ", "42"},
		{"Yes", "yes"},
	}

	for _, tt := range tests {
		got := NormalizeAnswer(tt.input)
		if got != tt.want {
			t.Errorf("NormalizeAnswer(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestEvaluateConsensus(t *testing.T) {
	candidates := []CandidateAnswer{
		{Index: 0, RawAnswer: "42", NormAnswer: "42", FullOutput: "Sol 1: 42"},
		{Index: 1, RawAnswer: "42.0", NormAnswer: "42", FullOutput: "Sol 2: 42.0"},
		{Index: 2, RawAnswer: "15", NormAnswer: "15", FullOutput: "Sol 3: 15"},
		{Index: 3, RawAnswer: "42", NormAnswer: "42", FullOutput: "Sol 4: 42"},
		{Index: 4, RawAnswer: "99", NormAnswer: "99", FullOutput: "Sol 5: 99"},
	}

	res := EvaluateConsensus(candidates)
	if res.WinningAnswer != "42" && res.WinningAnswer != "42.0" {
		t.Errorf("Expected winning answer 42, got %q", res.WinningAnswer)
	}
	if res.Votes != 3 {
		t.Errorf("Expected 3 votes, got %d", res.Votes)
	}
	if res.Confidence != 0.6 {
		t.Errorf("Expected 0.6 confidence, got %f", res.Confidence)
	}
	if res.BestCandidate == nil || res.BestCandidate.Index != 0 {
		t.Errorf("Expected candidate 0 as best candidate")
	}
}
