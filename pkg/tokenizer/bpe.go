package tokenizer

import (
	"fmt"
	"strconv"
	"strings"
)

// Tokenizer implements a Byte-Pair Encoding (BPE) tokenizer compatible with GGUF models.
type Tokenizer struct {
	Vocab         []string
	TokenToID     map[string]int
	Merges        map[string]int
	Scores        []float32
	BosTokenID    int
	EosTokenID    int
	EotTokenID    int
	SpecialTokens map[string]int
	ByteFallback  bool
}

// NewTokenizer constructs a Tokenizer instance from vocabulary and merge rules.
func NewTokenizer(vocab []string, merges []string, bosID, eosID int) *Tokenizer {
	t2i := make(map[string]int, len(vocab))
	special := make(map[string]int)

	for id, token := range vocab {
		t2i[token] = id
		if strings.HasPrefix(token, "<|") && strings.HasSuffix(token, "|>") {
			special[token] = id
		} else if strings.HasPrefix(token, "<") && strings.HasSuffix(token, ">") && !strings.HasPrefix(token, "<0x") {
			special[token] = id
		}
	}

	mergeMap := make(map[string]int, len(merges))
	for rank, m := range merges {
		mergeMap[m] = rank
	}

	eotID := -1
	if id, ok := t2i["<|eot_id|>"]; ok {
		eotID = id
	} else if id, ok := t2i["<|im_end|>"]; ok {
		eotID = id
	} else {
		eotID = eosID
	}

	return &Tokenizer{
		Vocab:         vocab,
		TokenToID:     t2i,
		Merges:        mergeMap,
		BosTokenID:    bosID,
		EosTokenID:    eosID,
		EotTokenID:    eotID,
		SpecialTokens: special,
		ByteFallback:  true,
	}
}

// Encode converts raw text into token IDs via BPE merges.
func (t *Tokenizer) Encode(text string, addBos bool) []int {
	var tokens []int
	if addBos && t.BosTokenID >= 0 {
		tokens = append(tokens, t.BosTokenID)
	}

	if len(text) == 0 {
		return tokens
	}

	// Split text by special tokens if present
	segments := t.splitSpecialTokens(text)
	for _, seg := range segments {
		if seg.isSpecial {
			tokens = append(tokens, seg.tokenID)
			continue
		}

		segTokens := t.encodeSegment(seg.text)
		tokens = append(tokens, segTokens...)
	}

	return tokens
}

type segment struct {
	text      string
	isSpecial bool
	tokenID   int
}

func (t *Tokenizer) splitSpecialTokens(text string) []segment {
	var result []segment
	remaining := text

	for len(remaining) > 0 {
		foundSpecial := false
		bestIdx := len(remaining)
		var bestToken string
		var bestID int

		for tokenStr, id := range t.SpecialTokens {
			idx := strings.Index(remaining, tokenStr)
			if idx != -1 && idx < bestIdx {
				bestIdx = idx
				bestToken = tokenStr
				bestID = id
				foundSpecial = true
			}
		}

		if !foundSpecial {
			result = append(result, segment{text: remaining, isSpecial: false})
			break
		}

		if bestIdx > 0 {
			result = append(result, segment{text: remaining[:bestIdx], isSpecial: false})
		}
		result = append(result, segment{text: bestToken, isSpecial: true, tokenID: bestID})
		remaining = remaining[bestIdx+len(bestToken):]
	}

	return result
}

func (t *Tokenizer) encodeSegment(text string) []int {
	if len(text) == 0 {
		return nil
	}

	var tokens []int
	// Initial tokenization at byte level or individual characters
	for i := 0; i < len(text); i++ {
		b := text[i : i+1]
		if id, ok := t.TokenToID[b]; ok {
			tokens = append(tokens, id)
		} else if t.ByteFallback {
			byteToken := fmt.Sprintf("<0x%02X>", text[i])
			if id, ok := t.TokenToID[byteToken]; ok {
				tokens = append(tokens, id)
			} else {
				// Fallback to unknown or raw byte
				tokens = append(tokens, int(text[i]))
			}
		}
	}

	// Iteratively apply BPE merges based on lowest rank
	for {
		if len(tokens) < 2 {
			break
		}

		bestRank := int(^uint(0) >> 1)
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
			if tokens[i] < 0 || tokens[i] >= len(t.Vocab) || tokens[i+1] < 0 || tokens[i+1] >= len(t.Vocab) {
				continue
			}
			t1 := t.Vocab[tokens[i]]
			t2 := t.Vocab[tokens[i+1]]
			pair := t1 + " " + t2

			if rank, exists := t.Merges[pair]; exists && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break // No further merges possible
		}

		mergedStr := t.Vocab[tokens[bestIdx]] + t.Vocab[tokens[bestIdx+1]]
		if mergedID, exists := t.TokenToID[mergedStr]; exists {
			newTokens := make([]int, 0, len(tokens)-1)
			newTokens = append(newTokens, tokens[:bestIdx]...)
			newTokens = append(newTokens, mergedID)
			newTokens = append(newTokens, tokens[bestIdx+2:]...)
			tokens = newTokens
		} else {
			break
		}
	}

	return tokens
}

// Decode converts token IDs back to a readable UTF-8 string.
func (t *Tokenizer) Decode(tokens []int) string {
	var rawBytes []byte
	for _, tok := range tokens {
		if tok < 0 || tok >= len(t.Vocab) {
			continue
		}
		piece := t.Vocab[tok]

		// Handle byte tokens <0xXX>
		if strings.HasPrefix(piece, "<0x") && strings.HasSuffix(piece, ">") && len(piece) == 6 {
			if hexVal, err := strconv.ParseUint(piece[3:5], 16, 8); err == nil {
				rawBytes = append(rawBytes, byte(hexVal))
				continue
			}
		}

		// Replace special space symbols: ' ' (\u2581) or 'Ġ' (\u0120)
		cleaned := strings.ReplaceAll(piece, " ", " ")
		cleaned = strings.ReplaceAll(cleaned, "Ġ", " ")
		rawBytes = append(rawBytes, []byte(cleaned)...)
	}

	return string(rawBytes)
}

// DecodeToken decodes a single token into its string representation.
func (t *Tokenizer) DecodeToken(tok int) string {
	return t.Decode([]int{tok})
}
