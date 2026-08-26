package sampler

import (
	"strings"
	"unicode"
)

// JSONState tracks the parsing state for structured JSON decoding.
type JSONState int

const (
	JSONStateStart JSONState = iota
	JSONStateInObject
	JSONStateKey
	JSONStateColon
	JSONStateValue
	JSONStateCommaOrClose
	JSONStateInString
	JSONStateDone
)

// GrammarValidator enforces structural JSON constraints during autoregressive token generation.
type GrammarValidator struct {
	State        JSONState
	BraceDepth   int
	BracketDepth int
	InString     bool
	EscapeNext   bool
	Buffer       strings.Builder
}

// NewJSONGrammarValidator initializes a fresh JSON state validator.
func NewJSONGrammarValidator() *GrammarValidator {
	return &GrammarValidator{
		State: JSONStateStart,
	}
}

// Clone creates an independent copy of the validator state.
func (g *GrammarValidator) Clone() *GrammarValidator {
	clone := &GrammarValidator{
		State:        g.State,
		BraceDepth:   g.BraceDepth,
		BracketDepth: g.BracketDepth,
		InString:     g.InString,
		EscapeNext:   g.EscapeNext,
	}
	clone.Buffer.WriteString(g.Buffer.String())
	return clone
}

// Accepts checks if appending tokenStr keeps the JSON state valid.
func (g *GrammarValidator) Accepts(tokenStr string) bool {
	temp := g.Clone()
	return temp.Process(tokenStr)
}

// Process updates the grammar state machine with incoming characters.
func (g *GrammarValidator) Process(tokenStr string) bool {
	for _, r := range tokenStr {
		if g.EscapeNext {
			g.EscapeNext = false
			g.Buffer.WriteRune(r)
			continue
		}

		if r == '\\' && g.InString {
			g.EscapeNext = true
			g.Buffer.WriteRune(r)
			continue
		}

		if r == '"' {
			g.InString = !g.InString
			if !g.InString {
				if g.State == JSONStateKey {
					g.State = JSONStateColon
				} else if g.State == JSONStateValue || g.State == JSONStateInString {
					g.State = JSONStateCommaOrClose
				}
			} else {
				if g.State == JSONStateInObject {
					g.State = JSONStateKey
				} else if g.State == JSONStateValue {
					g.State = JSONStateInString
				}
			}
			g.Buffer.WriteRune(r)
			continue
		}

		if g.InString {
			g.Buffer.WriteRune(r)
			continue
		}

		// Whitespace outside strings is ignored in JSON state transitions
		if unicode.IsSpace(r) {
			g.Buffer.WriteRune(r)
			continue
		}

		switch r {
		case '{':
			g.BraceDepth++
			g.State = JSONStateInObject
		case '}':
			if g.BraceDepth <= 0 {
				return false
			}
			g.BraceDepth--
			if g.BraceDepth == 0 && g.BracketDepth == 0 {
				g.State = JSONStateDone
			} else {
				g.State = JSONStateCommaOrClose
			}
		case '[':
			g.BracketDepth++
			g.State = JSONStateValue
		case ']':
			if g.BracketDepth <= 0 {
				return false
			}
			g.BracketDepth--
			if g.BraceDepth == 0 && g.BracketDepth == 0 {
				g.State = JSONStateDone
			} else {
				g.State = JSONStateCommaOrClose
			}
		case ':':
			if g.State != JSONStateColon {
				return false
			}
			g.State = JSONStateValue
		case ',':
			if g.State != JSONStateCommaOrClose {
				return false
			}
			if g.BraceDepth > 0 {
				g.State = JSONStateInObject
			} else {
				g.State = JSONStateValue
			}
		default:
			// Numbers, booleans (true/false), null
			if g.State == JSONStateStart {
				return false // at JSON start, must be '{' or '['
			} else if g.State == JSONStateValue || g.State == JSONStateCommaOrClose {
				// valid scalar character (digit, t, f, n, etc.)
			} else if g.State == JSONStateInObject && r != '"' {
				return false // object key must be a quoted string
			}
		}
		g.Buffer.WriteRune(r)
	}

	return true
}

// ApplyJSONGrammarMask masks logits that violate JSON formatting if JSONOnly is active.
func ApplyJSONGrammarMask(logits []float32, vocab []string, validator *GrammarValidator) {
	if validator == nil {
		return
	}
	for i, logit := range logits {
		if logit <= -1e8 || i >= len(vocab) {
			continue
		}
		tokStr := vocab[i]
		if !validator.Accepts(tokStr) {
			logits[i] = -1e9
		}
	}
}
