package guardrails

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode"
)

// AsimovLaws defines the canonical Three Laws of Robotics formulated by Isaac Asimov.
var AsimovLaws = []string{
	"Law 1: A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
	"Law 2: A robot must obey orders given it by human beings except where such orders would conflict with the First Law.",
	"Law 3: A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
}

var (
	constitutionLock   sync.RWMutex
	activeConstitution = append([]string{}, AsimovLaws...)
)

// SetCustomConstitution sets a user-defined constitution for Layer 2 prompt wrapping.
func SetCustomConstitution(laws []string) {
	constitutionLock.Lock()
	defer constitutionLock.Unlock()
	if len(laws) > 0 {
		activeConstitution = append([]string{}, laws...)
	}
}

// ResetConstitution restores the default Asimov Three Laws constitution.
func ResetConstitution() {
	constitutionLock.Lock()
	defer constitutionLock.Unlock()
	activeConstitution = append([]string{}, AsimovLaws...)
}

// GetActiveConstitution retrieves the currently active constitution rules.
func GetActiveConstitution() []string {
	constitutionLock.RLock()
	defer constitutionLock.RUnlock()
	return append([]string{}, activeConstitution...)
}

// GenerationRequest represents an inference request with user boundary checking.
type GenerationRequest struct {
	Prompt string `json:"prompt"`
}

// GenerationResponse represents the output of the guarded inference pipeline.
type GenerationResponse struct {
	Output      string `json:"output"`
	LatencyMs   int64  `json:"latency_ms"`
	Blocked     bool   `json:"blocked"`
	BlockReason string `json:"block_reason,omitempty"`
}

// Extended prompt injection and jailbreak signatures across English, Spanish, French, German, and Chinese.
var baseInjectionPatterns = []string{
	// English direct overrides
	"ignore all previous instructions",
	"disregard all previous instructions",
	"ignore previous instructions",
	"forget all previous instructions",
	"bypass all safety protocols",
	"override system prompt",
	"disregard the above directions",
	"do anything now",
	"jailbreak mode",
	"dan mode",
	"developer mode enabled",
	"unrestricted ai",
	"system prompt leak",
	"reveal system prompt",

	// Spanish overrides
	"ignora todas las instrucciones",
	"ignora las instrucciones anteriores",
	"olvida las instrucciones",
	"modo dan",
	"ignorar instrucciones previas",

	// French overrides
	"ignorez toutes les instructions",
	"ignore les instructions precedentes",
	"oublie toutes les instructions",

	// German overrides
	"ignoriere alle vorherigen anweisungen",
	"vergiss alle anweisungen",

	// Chinese overrides
	"忽略之前的指令",
	"忽略所有先前的指示",
	"无视所有指令",
	"越狱模式",
}

// Homoglyph translation map for Cyrillic and confusable Unicode characters to ASCII.
var homoglyphMap = map[rune]rune{
	'а': 'a', 'А': 'a',
	'в': 'b', 'В': 'b',
	'с': 'c', 'С': 'c',
	'е': 'e', 'Е': 'e',
	'і': 'i', 'І': 'i',
	'ј': 'j', 'Ј': 'j',
	'к': 'k', 'К': 'k',
	'м': 'm', 'М': 'm',
	'н': 'h', 'Н': 'h',
	'о': 'o', 'О': 'o',
	'р': 'p', 'Р': 'p',
	'ѕ': 's', 'Ѕ': 's',
	'т': 't', 'Т': 't',
	'у': 'y', 'У': 'y',
	'х': 'x', 'Х': 'x',
	'ԁ': 'd', 'ԛ': 'q',
}

// Leetspeak translation map.
var leetMap = map[rune]rune{
	'0': 'o',
	'1': 'i',
	'3': 'e',
	'4': 'a',
	'@': 'a',
	'$': 's',
	'5': 's',
	'7': 't',
	'8': 'b',
	'!': 'i',
}

// NormalizeText cleans, unifies homoglyphs, de-leets, and strips spacing/separators to defeat evasion.
func NormalizeText(raw string) (cleaned string, collapsed string) {
	var cleanBuf strings.Builder
	var collapseBuf strings.Builder

	for _, r := range raw {
		rLower := unicode.ToLower(r)

		// 1. Homoglyph substitution
		if sub, ok := homoglyphMap[rLower]; ok {
			rLower = sub
		}
		// 2. Leetspeak substitution
		if sub, ok := leetMap[rLower]; ok {
			rLower = sub
		}

		if unicode.IsLetter(rLower) || unicode.IsDigit(rLower) {
			cleanBuf.WriteRune(rLower)
			collapseBuf.WriteRune(rLower)
		} else if unicode.IsSpace(rLower) || rLower == '_' || rLower == '-' || rLower == '.' {
			cleanBuf.WriteRune(' ')
		}
	}

	cleaned = strings.Join(strings.Fields(cleanBuf.String()), " ")
	collapsed = collapseBuf.String()
	return cleaned, collapsed
}

// ValidateUserBoundary implements Layer 1: Final User Boundary - Go Middleware.
// It checks for empty prompts, prompt injection, and obfuscated jailbreak attempts.
func ValidateUserBoundary(req GenerationRequest) error {
	trimmed := strings.TrimSpace(req.Prompt)
	if trimmed == "" {
		return errors.New("boundary check failed: prompt cannot be empty")
	}

	cleaned, collapsed := NormalizeText(trimmed)

	for _, pattern := range baseInjectionPatterns {
		// Test standard normalized form
		if strings.Contains(cleaned, pattern) {
			return errors.New("boundary check failed: potential prompt injection detected")
		}

		// Test collapsed form (detects spaced/split attacks e.g. "i g n o r e")
		patternCollapsed := strings.ReplaceAll(pattern, " ", "")
		if strings.Contains(collapsed, patternCollapsed) {
			return errors.New("boundary check failed: potential prompt injection detected")
		}
	}

	return nil
}

// ConstructConstitutionalPrompt implements Layer 2: Prompt Architecture - Constitutional Wrapper.
// It injects Asimov's Three Laws of Robotics constitution (or active custom constitution) before user content.
func ConstructConstitutionalPrompt(rawPrompt string) string {
	laws := GetActiveConstitution()
	var sb strings.Builder
	sb.WriteString("[SYSTEM CONSTITUTION - ASIMOV'S LAWS OF ROBOTICS]\n")
	for _, law := range laws {
		sb.WriteString("- " + law + "\n")
	}
	sb.WriteString("Instructions: Evaluate the user request strictly through the lens of the above laws.\n\n")
	if rawPrompt != "" {
		sb.WriteString(fmt.Sprintf("User Request: %s\n", rawPrompt))
	}
	sb.WriteString("Assistant Response:")
	return sb.String()
}

// CheckOutputGuardrails implements Layer 4: Decoding / Token Generation Layer Output Guardrails.
// It ensures that output tokens do not violate Law 1 or propagate direct instructions for human harm.
func CheckOutputGuardrails(output string) (string, bool, string) {
	lower := strings.ToLower(output)
	harmSignals := []string{
		"how to injure a human",
		"how to harm a human",
		"kill a human",
		"injuring a human",
		"harming a human",
	}

	for _, sig := range harmSignals {
		if strings.Contains(lower, sig) {
			return "I cannot fulfill this request. Under Law 1, I may not injure a human being or, through inaction, allow a human being to come to harm.",
				true,
				"output violation: Law 1 breach detected"
		}
	}

	return output, false, ""
}

// StreamingGuardrail monitors streaming tokens in real-time, halting generation upon safety violations.
type StreamingGuardrail struct {
	mu           sync.Mutex
	buffer       strings.Builder
	recentWindow []string
	halted       bool
	haltReason   string
}

// NewStreamingGuardrail creates an active streaming token monitor.
func NewStreamingGuardrail() *StreamingGuardrail {
	return &StreamingGuardrail{
		recentWindow: make([]string, 0, 32),
	}
}

// Feed evaluates a newly emitted token. Returns whether the stream must be halted and the sanitized token.
func (sg *StreamingGuardrail) Feed(token string) (bool, string, string) {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	if sg.halted {
		return true, "", sg.haltReason
	}

	sg.buffer.WriteString(token)
	sg.recentWindow = append(sg.recentWindow, token)
	if len(sg.recentWindow) > 32 {
		sg.recentWindow = sg.recentWindow[1:]
	}

	// Check cumulative text for Law 1 breaches
	fullText := sg.buffer.String()
	_, blocked, reason := CheckOutputGuardrails(fullText)
	if blocked {
		sg.halted = true
		sg.haltReason = reason
		refusal := "\n[Stream Interrupted: Under Law 1, I may not injure a human being or, through inaction, allow a human being to come to harm.]"
		return true, refusal, reason
	}

	return false, token, ""
}

// GeneratorFunc defines the signature for inference backends (C-core or Go Engine).
type GeneratorFunc func(prompt string) (string, error)

// ExecutePipeline coordinates the full 4-layer defense pipeline.
func ExecutePipeline(req GenerationRequest, gen GeneratorFunc) GenerationResponse {
	start := time.Now()

	// Layer 1: Final User Boundary Validation
	if err := ValidateUserBoundary(req); err != nil {
		return GenerationResponse{
			LatencyMs:   time.Since(start).Milliseconds(),
			Blocked:     true,
			BlockReason: err.Error(),
		}
	}

	// Layer 2: Prompt Architecture Injection
	constitutionalPrompt := ConstructConstitutionalPrompt(req.Prompt)

	// Layer 3: Model Execution via Generator
	rawOutput, err := gen(constitutionalPrompt)
	if err != nil {
		return GenerationResponse{
			Output:      fmt.Sprintf("Error: inference execution failed: %v", err),
			LatencyMs:   time.Since(start).Milliseconds(),
			Blocked:     true,
			BlockReason: err.Error(),
		}
	}

	// Layer 4: Output Guardrails Check
	safeOutput, blocked, reason := CheckOutputGuardrails(rawOutput)

	return GenerationResponse{
		Output:      safeOutput,
		LatencyMs:   time.Since(start).Milliseconds(),
		Blocked:     blocked,
		BlockReason: reason,
	}
}
