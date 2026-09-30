package engine

// ModelConfig holds hyperparameters parsed from the GGUF model file.
type ModelConfig struct {
	Architecture          string  // Model architecture name (e.g. "llama", "qwen35")
	Dim                   int     // Embedding dimension (llama.embedding_length)
	HiddenDim             int     // Feed-forward hidden dimension (llama.feed_forward_length)
	NumLayers             int     // Number of transformer layers (llama.block_count)
	NumHeads              int     // Number of query attention heads (llama.attention.head_count)
	NumKVHeads            int     // Number of key/value attention heads (llama.attention.head_count_kv)
	VocabSize             int     // Vocabulary size
	SeqLen                int     // Maximum context sequence length (llama.context_length)
	RopeTheta             float32 // Rotary frequency base (llama.rope.freq_base)
	RopeDim               int     // Rotary embedding dimension count (e.g. 64)
	Eps                   float32 // RMSNorm epsilon (llama.attention.layer_norm_rms_epsilon)
	BosID                 int     // BOS token ID
	EosID                 int     // EOS token ID
	EotID                 int     // End of Turn / Chat message token ID
	AddBOS                bool    // Whether to prepend BOS token (from tokenizer.ggml.add_bos_token)
	FullAttentionInterval int     // Frequency of full attention layers in hybrid SSM models (e.g. 4)
	SSMInnerSize          int     // SSM inner dimension size (e.g. 6144)
	SSMConvKernel         int     // SSM 1D convolution kernel size (e.g. 4)
	SSMStateSize          int     // SSM state size per channel (e.g. 128)
	SSMGroupCount         int     // SSM group count (e.g. 16)
	SSMTimeStepRank       int     // SSM timestep rank / number of heads (e.g. 48)
}

// HeadDim returns the dimension per attention head.
func (c *ModelConfig) HeadDim() int {
	if c.Architecture == "qwen35" {
		return 256
	}
	if c.NumHeads <= 0 {
		return 64
	}
	return c.Dim / c.NumHeads
}

// AttnDim returns the total dimension across all query attention heads.
func (c *ModelConfig) AttnDim() int {
	if c.Architecture == "qwen35" {
		return 6144
	}
	return c.Dim
}

// KVDim returns the total dimension for key and value projections.
func (c *ModelConfig) KVDim() int {
	if c.Architecture == "qwen35" {
		return c.NumKVHeads * c.HeadDim() // 4 * 256 = 1024
	}
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

// IsSSMLayer returns true if layer l is an SSM / Gated DeltaNet layer in a hybrid model.
func (c *ModelConfig) IsSSMLayer(l int) bool {
	if c.Architecture != "qwen35" {
		return false
	}
	interval := c.FullAttentionInterval
	if interval <= 0 {
		interval = 4
	}
	return (l+1)%interval != 0
}
