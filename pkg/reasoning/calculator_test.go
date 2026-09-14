package reasoning

import (
	"math"
	"testing"
)

func TestEvaluateMath(t *testing.T) {
	tests := []struct {
		expr     string
		wantVal  float64
		wantStr  string
		wantErr  bool
	}{
		// Basic Arithmetic
		{"1 + 1", 2, "2", false},
		{"10 - 4", 6, "6", false},
		{"3 * 4", 12, "12", false},
		{"15 / 3", 5, "5", false},
		{"17 % 5", 2, "2", false},
		{"2 + 3 * 4", 14, "14", false},
		{"(2 + 3) * 4", 20, "20", false},

		// Powers and Exponents
		{"2 ^ 10", 1024, "1024", false},
		{"2 ** 8", 256, "256", false},
		{"2 ^ 3 ^ 2", 512, "512", false}, // right associative: 2^(3^2) = 2^9 = 512

		// Unary and Postfix Factorial
		{"-5 + 10", 5, "5", false},
		{"5!", 120, "120", false},
		{"0!", 1, "1", false},
		{"3! * 2", 12, "12", false},

		// Functions
		{"sqrt(144)", 12, "12", false},
		{"cbrt(27)", 3, "3", false},
		{"abs(-42)", 42, "42", false},
		{"floor(3.7)", 3, "3", false},
		{"ceil(3.2)", 4, "4", false},
		{"round(3.6)", 4, "4", false},
		{"max(10, 25, 5)", 25, "25", false},
		{"min(10, 25, 5)", 5, "5", false},
		{"pow(3, 3)", 27, "27", false},
		{"log(100)", 2, "2", false},
		{"exp(0)", 1, "1", false},

		// Constants
		{"pi * 2", math.Pi * 2, FormatResult(math.Pi * 2), false},
		{"e", math.E, FormatResult(math.E), false},

		// Decimals and floats
		{"0.1 + 0.2", 0.3, "0.3", false},
		{"1.5 * 4", 6, "6", false},

		// Complex chained math problem
		{"(125 / 5) + 3^3 - 10", 42, "42", false},

		// Error cases
		{"1 / 0", 0, "", true},
		{"5 % 0", 0, "", true},
		{"sqrt(-4)", 0, "", true},
		{"(-2)!", 0, "", true},
		{"(1 + 2", 0, "", true},
		{"unknown_func(5)", 0, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.expr, func(t *testing.T) {
			val, str, err := EvaluateMath(tt.expr)
			if (err != nil) != tt.wantErr {
				t.Fatalf("EvaluateMath(%q) error = %v, wantErr %v", tt.expr, err, tt.wantErr)
			}
			if !tt.wantErr {
				if math.Abs(val-tt.wantVal) > 1e-9 {
					t.Errorf("EvaluateMath(%q) val = %g, want %g", tt.expr, val, tt.wantVal)
				}
				if str != tt.wantStr {
					t.Errorf("EvaluateMath(%q) str = %q, want %q", tt.expr, str, tt.wantStr)
				}
			}
		})
	}
}
