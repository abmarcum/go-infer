package reasoning

import (
	"strings"
	"testing"
)

func TestExecuteMathTool(t *testing.T) {
	// Direct expression
	res, err := ExecuteMathTool("15 * 4 + 2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res != "62" {
		t.Errorf("got %q, want 62", res)
	}

	// JSON payload
	jsonPayload := `{"expression": "sqrt(256) / 2"}`
	resJSON, err := ExecuteMathTool(jsonPayload)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resJSON != "8" {
		t.Errorf("got %q, want 8", resJSON)
	}
}

func TestProcessInlineMath(t *testing.T) {
	inputCodeBlock := "Let's calculate the perimeter:\n```calc\n2 * (10 + 15)\n```\nThus the perimeter is found."
	out, modified := ProcessInlineMath(inputCodeBlock)
	if !modified {
		t.Fatalf("expected modified = true")
	}
	if !strings.Contains(out, "--> result: 50") {
		t.Errorf("expected result 50 in output, got:\n%s", out)
	}

	inputTag := "The value is <<calc: 3^4>> units."
	outTag, modTag := ProcessInlineMath(inputTag)
	if !modTag {
		t.Fatalf("expected modTag = true")
	}
	if !strings.Contains(outTag, "<<calc: 3^4 = 81>>") {
		t.Errorf("expected <<calc: 3^4 = 81>>, got:\n%s", outTag)
	}
}
