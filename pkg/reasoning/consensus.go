package reasoning

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// CandidateAnswer represents a single completion candidate in self-consistency voting.
type CandidateAnswer struct {
	Index        int    `json:"index"`
	RawAnswer    string `json:"raw_answer"`
	NormAnswer   string `json:"normalized_answer"`
	Thinking     string `json:"thinking,omitempty"`
	FullOutput   string `json:"full_output"`
}

// ConsensusResult represents the aggregated result of self-consistency majority voting.
type ConsensusResult struct {
	WinningAnswer  string            `json:"winning_answer"`
	NormAnswer     string            `json:"normalized_answer"`
	Confidence     float64           `json:"confidence"`
	Votes          int               `json:"votes"`
	TotalSamples   int               `json:"total_samples"`
	BestCandidate  *CandidateAnswer  `json:"best_candidate"`
	Distribution   map[string]int    `json:"distribution"`
	AllCandidates  []CandidateAnswer `json:"all_candidates"`
}

var (
	boxedRegex        = regexp.MustCompile(`\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}`)
	gsm8kRegex        = regexp.MustCompile(`(?m)####\s*([^\n\r]+)`)
	finalAnswerRegex  = regexp.MustCompile(`(?i)(?:final\s+answer|the\s+answer\s+is|answer:)\s*[:=]?\s*([^\n\r]+)`)
	boldAnswerRegex   = regexp.MustCompile(`\*\*\$?([0-9]+(?:\.[0-9]+)?)\*\*`)
	dollarAnswerRegex = regexp.MustCompile(`(?i)(?:makes|earns|total|received|got|left|is|equals)?\s*\$([0-9]+(?:\.[0-9]+)?)`)
	fracRegex         = regexp.MustCompile(`\\frac\{([^{}]+)\}\{([^{}]+)\}`)
)

// ExtractAnswer extracts the primary mathematical answer from model output text.
func ExtractAnswer(text string) string {
	clean := strings.TrimSpace(text)
	if clean == "" {
		return ""
	}

	// 1. Check for LaTeX \boxed{...} with balanced brace parser
	if idx := strings.Index(clean, `\boxed{`); idx != -1 {
		start := idx + len(`\boxed{`)
		depth := 1
		for i := start; i < len(clean); i++ {
			if clean[i] == '{' {
				depth++
			} else if clean[i] == '}' {
				depth--
				if depth == 0 {
					return strings.TrimSpace(clean[start:i])
				}
			}
		}
	}

	// 2. Check for GSM8K standard delimiter "#### <answer>"
	if m := gsm8kRegex.FindStringSubmatch(clean); len(m) > 1 {
		return strings.Trim(strings.TrimSpace(m[1]), " .\t\n\r")
	}

	// 3. Check for "Final Answer: <answer>" or "The answer is: <answer>"
	if m := finalAnswerRegex.FindStringSubmatch(clean); len(m) > 1 {
		return strings.Trim(strings.TrimSpace(m[1]), " .\t\n\r")
	}

	// 4. Check for Markdown bolded numeric answer e.g. **$18** or **18**
	if matches := boldAnswerRegex.FindAllStringSubmatch(clean, -1); len(matches) > 0 {
		lastMatch := matches[len(matches)-1]
		if len(lastMatch) > 1 {
			return lastMatch[1]
		}
	}

	// 5. Fallback: Check the concluding lines
	lines := strings.Split(clean, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		if l == "" || strings.HasPrefix(l, "</think>") || l == `\\` || l == `\` {
			continue
		}
		// If line has a dollar amount like $18 or \$18
		if m := dollarAnswerRegex.FindStringSubmatch(l); len(m) > 1 {
			return m[1]
		}
		return strings.Trim(l, " .\t\n\r")
	}

	return clean
}

// NormalizeAnswer normalizes mathematical expressions and numbers for semantic equivalence comparison.
func NormalizeAnswer(ans string) string {
	s := strings.TrimSpace(ans)
	if s == "" {
		return ""
	}

	// Strip \boxed{...} wrapper if present
	if strings.HasPrefix(s, `\boxed{`) && strings.HasSuffix(s, "}") {
		s = s[len(`\boxed{`) : len(s)-1]
		s = strings.TrimSpace(s)
	}

	// Convert LaTeX fractions \frac{a}{b} -> a/b
	if fracRegex.MatchString(s) {
		s = fracRegex.ReplaceAllString(s, "$1/$2")
	}

	// Strip LaTeX wrappers e.g. \text{...}, \mathbf{...}, $, etc.
	s = strings.Trim(s, "$*\\ \t\n\r")
	s = strings.ReplaceAll(s, `\text{`, "")
	s = strings.ReplaceAll(s, `\mathbf{`, "")
	s = strings.ReplaceAll(s, `\mathrm{`, "")
	s = strings.ReplaceAll(s, `**`, "")
	s = strings.Trim(s, "{}")

	// Strip currency and units
	s = strings.TrimPrefix(s, "$")
	s = strings.TrimSuffix(s, "%")
	s = strings.TrimSuffix(s, " dollars")
	s = strings.TrimSuffix(s, " miles")

	// Remove commas in numbers like 1,000 -> 1000
	cleanNum := strings.ReplaceAll(s, ",", "")
	cleanNum = strings.TrimSpace(cleanNum)

	if cleanNum == "\\" || cleanNum == "=" || cleanNum == ":" || cleanNum == "" {
		return ""
	}

	// Check if fraction a/b
	if strings.Contains(cleanNum, "/") {
		parts := strings.Split(cleanNum, "/")
		if len(parts) == 2 {
			num, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
			den, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
			if err1 == nil && err2 == nil && den != 0 {
				val := num / den
				return formatNumericNorm(val)
			}
		}
	}

	// Try parsing as float
	if val, err := strconv.ParseFloat(cleanNum, 64); err == nil {
		return formatNumericNorm(val)
	}

	// Lowercase and trim punctuation
	s = strings.ToLower(s)
	s = strings.Trim(s, ".:;!?'\"")
	return strings.TrimSpace(s)
}

func formatNumericNorm(val float64) string {
	rounded := math.Round(val)
	if math.Abs(val-rounded) < 1e-9 {
		return fmt.Sprintf("%.0f", rounded)
	}
	return strconv.FormatFloat(val, 'g', 8, 64)
}

// EvaluateConsensus runs majority voting over a set of completion candidate outputs.
func EvaluateConsensus(candidates []CandidateAnswer) *ConsensusResult {
	if len(candidates) == 0 {
		return &ConsensusResult{
			Distribution: make(map[string]int),
		}
	}

	freq := make(map[string]int)
	rawMap := make(map[string]string)
	candMap := make(map[string]*CandidateAnswer)

	for i := range candidates {
		c := &candidates[i]
		if c.NormAnswer == "" {
			c.NormAnswer = NormalizeAnswer(c.RawAnswer)
		}
		norm := c.NormAnswer
		if norm == "" {
			norm = "<empty>"
		}
		freq[norm]++
		if _, exists := rawMap[norm]; !exists {
			rawMap[norm] = c.RawAnswer
			candMap[norm] = c
		}
	}

	// Sort by frequency descending
	type answerTally struct {
		norm  string
		count int
	}
	var tallies []answerTally
	for k, v := range freq {
		tallies = append(tallies, answerTally{norm: k, count: v})
	}
	// Sort by frequency descending; prefer non-empty answers over <empty>
	sort.Slice(tallies, func(i, j int) bool {
		if tallies[i].count != tallies[j].count {
			return tallies[i].count > tallies[j].count
		}
		if tallies[i].norm == "<empty>" {
			return false
		}
		if tallies[j].norm == "<empty>" {
			return true
		}
		return tallies[i].norm < tallies[j].norm
	})

	winnerNorm := tallies[0].norm
	winnerVotes := tallies[0].count

	// If the top tally is <empty> but other valid answers exist, choose the top valid answer
	if winnerNorm == "<empty>" && len(tallies) > 1 {
		for _, t := range tallies {
			if t.norm != "<empty>" {
				winnerNorm = t.norm
				winnerVotes = t.count
				break
			}
		}
	}

	winnerRaw := rawMap[winnerNorm]
	bestCand := candMap[winnerNorm]

	confidence := float64(winnerVotes) / float64(len(candidates))

	return &ConsensusResult{
		WinningAnswer:  winnerRaw,
		NormAnswer:     winnerNorm,
		Confidence:     confidence,
		Votes:          winnerVotes,
		TotalSamples:   len(candidates),
		BestCandidate:  bestCand,
		Distribution:   freq,
		AllCandidates:  candidates,
	}
}
