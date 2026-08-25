package tokenizer

import (
	"testing"
)

func TestTokenizerEncodeDecode(t *testing.T) {
	vocab := []string{
		"<unk>", "<s>", "</s>",
		"h", "e", "l", "o", "w", "r", "d", " ",
		"he", "ll", "o", "world", "hello",
		"<|eot_id|>",
	}
	merges := []string{
		"h e",
		"l l",
		"he ll",
		"hell o",
	}

	tok := NewTokenizer(vocab, merges, 1, 2)

	text := "hello"
	tokens := tok.Encode(text, false)
	if len(tokens) == 0 {
		t.Fatalf("Expected encoded tokens, got empty")
	}

	decoded := tok.Decode(tokens)
	if decoded != text {
		t.Errorf("Decode mismatch: got %q, expected %q", decoded, text)
	}
}

func TestTokenizerSpecialTokens(t *testing.T) {
	vocab := []string{
		"<unk>", "<s>", "</s>", "<|eot_id|>", "a", "b",
	}
	merges := []string{}
	tok := NewTokenizer(vocab, merges, 1, 2)

	text := "a<|eot_id|>b"
	tokens := tok.Encode(text, false)
	if len(tokens) != 3 {
		t.Fatalf("Expected 3 tokens, got %d (%v)", len(tokens), tokens)
	}
	if tokens[1] != 3 {
		t.Errorf("Expected token[1] to be special token ID 3 (<|eot_id|>), got %d", tokens[1])
	}
}
