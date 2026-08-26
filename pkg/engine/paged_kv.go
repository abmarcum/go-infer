package engine

import (
	"fmt"
	"sync"
)

const (
	DefaultBlockTokens = 16
)

// KVBlock stores attention Key and Value vectors for a fixed window of tokens across all layers.
type KVBlock struct {
	ID        int
	NumLayers int
	BlockSize int
	KVDim     int
	Key       [][]float32 // [layer][token_idx * kv_dim]
	Value     [][]float32 // [layer][token_idx * kv_dim]
}

// NewKVBlock allocates a physical KV block for DefaultBlockTokens.
func NewKVBlock(id, numLayers, kvDim, blockSize int) *KVBlock {
	key := make([][]float32, numLayers)
	val := make([][]float32, numLayers)
	for l := 0; l < numLayers; l++ {
		key[l] = make([]float32, blockSize*kvDim)
		val[l] = make([]float32, blockSize*kvDim)
	}
	return &KVBlock{
		ID:        id,
		NumLayers: numLayers,
		BlockSize: blockSize,
		KVDim:     kvDim,
		Key:       key,
		Value:     val,
	}
}

// BlockTable maps a logical sequence token position to physical block IDs.
type BlockTable struct {
	SequenceID int
	BlockIDs   []int
	NumTokens  int
	BlockSize  int
}

// PagedKVManager manages physical block allocation and sequence mapping (vLLM-style block allocator).
type PagedKVManager struct {
	mu         sync.Mutex
	NumLayers  int
	KVDim      int
	BlockSize  int
	TotalBlocks int
	FreeBlocks []int
	AllBlocks  []*KVBlock
	Tables     map[int]*BlockTable
	nextSeqID  int
}

// NewPagedKVManager initializes a paged KV memory pool.
func NewPagedKVManager(numLayers, kvDim, totalBlocks, blockSize int) *PagedKVManager {
	if blockSize <= 0 {
		blockSize = DefaultBlockTokens
	}
	if totalBlocks <= 0 {
		totalBlocks = 256
	}

	freeBlocks := make([]int, totalBlocks)
	allBlocks := make([]*KVBlock, totalBlocks)
	for i := 0; i < totalBlocks; i++ {
		freeBlocks[i] = i
		allBlocks[i] = NewKVBlock(i, numLayers, kvDim, blockSize)
	}

	return &PagedKVManager{
		NumLayers:   numLayers,
		KVDim:       kvDim,
		BlockSize:   blockSize,
		TotalBlocks: totalBlocks,
		FreeBlocks:  freeBlocks,
		AllBlocks:   allBlocks,
		Tables:      make(map[int]*BlockTable),
		nextSeqID:   1,
	}
}

// AllocateSequence registers a new sequence session and returns its sequence ID.
func (m *PagedKVManager) AllocateSequence() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	seqID := m.nextSeqID
	m.nextSeqID++

	m.Tables[seqID] = &BlockTable{
		SequenceID: seqID,
		BlockIDs:   make([]int, 0, 8),
		NumTokens:  0,
		BlockSize:  m.BlockSize,
	}
	return seqID
}

// AppendToken reserves physical block capacity for a new token in the sequence.
func (m *PagedKVManager) AppendToken(seqID int) (blockID, slotOffset int, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	table, ok := m.Tables[seqID]
	if !ok {
		return 0, 0, fmt.Errorf("sequence %d not found", seqID)
	}

	neededBlocks := (table.NumTokens / m.BlockSize) + 1
	if len(table.BlockIDs) < neededBlocks {
		if len(m.FreeBlocks) == 0 {
			return 0, 0, fmt.Errorf("out of paged KV memory (no free blocks available)")
		}
		// Allocate free block
		bID := m.FreeBlocks[len(m.FreeBlocks)-1]
		m.FreeBlocks = m.FreeBlocks[:len(m.FreeBlocks)-1]
		table.BlockIDs = append(table.BlockIDs, bID)
	}

	activeBlock := table.BlockIDs[len(table.BlockIDs)-1]
	slot := table.NumTokens % m.BlockSize
	table.NumTokens++

	return activeBlock, slot, nil
}

// FreeSequence returns all physical blocks of a sequence back to the free pool.
func (m *PagedKVManager) FreeSequence(seqID int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	table, ok := m.Tables[seqID]
	if !ok {
		return
	}

	for _, bID := range table.BlockIDs {
		m.FreeBlocks = append(m.FreeBlocks, bID)
	}
	delete(m.Tables, seqID)
}
