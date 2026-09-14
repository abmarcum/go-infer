package reasoning

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// CalculatorToolSchema returns an OpenAI-compatible function definition for mathematical calculation.
func CalculatorToolSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        "calculator",
			"description": "Safely evaluates mathematical expressions (arithmetic, exponents, powers, sqrt, abs, trigonometric, logarithms, factorial) with exact numerical precision.",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "The mathematical expression to evaluate, e.g. '(15 * 24) + sqrt(144)' or '2^10'",
					},
				},
				"required": []string{"expression"},
			},
		},
	}
}

// ExecuteMathTool parses a JSON arguments payload or direct string expression and returns the exact math result.
func ExecuteMathTool(argsOrExpr string) (string, error) {
	trimmed := strings.TrimSpace(argsOrExpr)
	if trimmed == "" {
		return "", fmt.Errorf("empty calculation input")
	}

	expr := trimmed
	// Try unmarshaling as JSON
	var payload struct {
		Expression string `json:"expression"`
		Expr       string `json:"expr"`
	}
	if err := json.Unmarshal([]byte(trimmed), &payload); err == nil {
		if payload.Expression != "" {
			expr = payload.Expression
		} else if payload.Expr != "" {
			expr = payload.Expr
		}
	}

	_, strVal, err := EvaluateMath(expr)
	if err != nil {
		return "", fmt.Errorf("calc error on '%s': %w", expr, err)
	}

	return strVal, nil
}

var (
	// Matches ```calc\n<expr>\n``` or ```math\n<expr>\n```
	calcCodeBlockRegex = regexp.MustCompile("(?s)```(?:calc|math)\n(.*?)\n```")
	// Matches <<calc: <expr>>> or <<math: <expr>>>
	inlineCalcTagRegex = regexp.MustCompile(`<<\s*(?:calc|math):\s*([^>]+)>>`)
)

// ProcessInlineMath scans text for embedded math execution markers, evaluates them, and appends the computed result.
func ProcessInlineMath(text string) (string, bool) {
	modified := false

	// Replace ```calc\n...\n``` with execution result
	res := calcCodeBlockRegex.ReplaceAllStringFunc(text, func(match string) string {
		sub := calcCodeBlockRegex.FindStringSubmatch(match)
		if len(sub) > 1 {
			expr := strings.TrimSpace(sub[1])
			if val, err := ExecuteMathTool(expr); err == nil {
				modified = true
				return fmt.Sprintf("```math\n%s\n--> result: %s\n```", expr, val)
			}
		}
		return match
	})

	// Replace <<calc: ...>> with <<calc: ... = result>>
	res = inlineCalcTagRegex.ReplaceAllStringFunc(res, func(match string) string {
		sub := inlineCalcTagRegex.FindStringSubmatch(match)
		if len(sub) > 1 {
			expr := strings.TrimSpace(sub[1])
			// Avoid re-evaluating if already has = result
			if strings.Contains(expr, "=") {
				return match
			}
			if val, err := ExecuteMathTool(expr); err == nil {
				modified = true
				return fmt.Sprintf("<<calc: %s = %s>>", expr, val)
			}
		}
		return match
	})

	return res, modified
}
