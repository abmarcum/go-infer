package reasoning

import (
	"go-inference/pkg/sampler"
	"math/rand"
	"strings"
	"time"
)

// ReasoningParams returns optimal sampling parameters recommended for reasoning models (DeepSeek-R1, QwQ, Qwen-Math).
func ReasoningParams() sampler.Params {
	return sampler.Params{
		Temperature: 0.6,
		TopP:        0.95,
		TopK:        40,
		RepPenalty:  1.0, // Disabled: rep penalty damages multi-step math derivations and symbol reuse
		Rand:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// ExtractThinking separates the <think>...</think> reasoning trace from the final solution.
func ExtractThinking(text string) (thinking string, answer string) {
	startIdx := strings.Index(text, "<think>")
	if startIdx == -1 {
		// If prompt already included <think>, generated text begins directly in the thought
		if endIdx := strings.Index(text, "</think>"); endIdx != -1 {
			thought := strings.TrimSpace(text[:endIdx])
			ans := strings.TrimSpace(text[endIdx+len("</think>"):])
			return thought, ans
		}
		return "", strings.TrimSpace(text)
	}

	contentStart := startIdx + len("<think>")
	endIdx := strings.Index(text[contentStart:], "</think>")
	if endIdx == -1 {
		// Thought is still in progress / unclosed
		return strings.TrimSpace(text[contentStart:]), ""
	}

	actualEnd := contentStart + endIdx
	thought := strings.TrimSpace(text[contentStart:actualEnd])
	ans := strings.TrimSpace(text[actualEnd+len("</think>"):])
	return thought, ans
}

// FormatThoughtDisplay formats thinking trace and answer for CLI/terminal display.
func FormatThoughtDisplay(thinking, answer string) string {
	var sb strings.Builder
	if thinking != "" {
		sb.WriteString("─── Thinking Process ───\n")
		sb.WriteString(thinking)
		sb.WriteString("\n────────────────────────\n\n")
	}
	sb.WriteString(answer)
	return sb.String()
}
