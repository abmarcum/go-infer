package engine

// ModelConfig holds hyperparameters parsed from the GGUF model file.
type ModelConfig struct {
	Dim        int     // Embedding dimension (llama.embedding_length)
	HiddenDim  int     // Feed-forward hidden dimension (llama.feed_forward_length)
	NumLayers  int     // Number of transformer layers (llama.block_count)
	NumHeads   int     // Number of query attention heads (llama.attention.head_count)
	NumKVHeads int     // Number of key/value attention heads (llama.attention.head_count_kv)
	VocabSize  int     // Vocabulary size
	SeqLen     int     // Maximum context sequence length (llama.context_length)
	RopeTheta  float32 // Rotary frequency base (llama.rope.freq_base)
	Eps        float32 // RMSNorm epsilon (llama.attention.layer_norm_rms_epsilon)
	BosID      int     // BOS token ID
	EosID      int     // EOS token ID
	EotID      int     // End of Turn / Chat message token ID
}

// HeadDim returns the dimension per attention head.
func (c *ModelConfig) HeadDim() int {
	if c.NumHeads <= 0 {
		return 64
	}
	return c.Dim / c.NumHeads
}

// KVDim returns the total dimension for key and value projections.
func (c *ModelConfig) KVDim() int {
	if c.NumHeads <= 0 {
		return c.Dim
	}
	return (c.Dim * c.NumKVHeads) / c.NumHeads
}

// KVMul returns the repeat factor for Grouped Query Attention (GQA).
func (c *ModelConfig) KVMul() int {
	if c.NumKVHeads <= 0 {
		return 1
	}
	return c.NumHeads / c.NumKVHeads
}
