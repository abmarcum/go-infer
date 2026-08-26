package engine

import (
	"testing"
)

func TestPagedKVManager(t *testing.T) {
	mgr := NewPagedKVManager(4, 64, 10, 16)

	// Allocate a sequence
	seq1 := mgr.AllocateSequence()
	if seq1 != 1 {
		t.Fatalf("Expected sequence ID 1, got %d", seq1)
	}

	// Append 35 tokens (should consume 3 blocks: 16 + 16 + 3)
	for i := 0; i < 35; i++ {
		bID, slot, err := mgr.AppendToken(seq1)
		if err != nil {
			t.Fatalf("Failed to append token %d: %v", i, err)
		}
		if bID < 0 || slot < 0 || slot >= 16 {
			t.Errorf("Invalid block/slot returned: (%d, %d)", bID, slot)
		}
	}

	table := mgr.Tables[seq1]
	if len(table.BlockIDs) != 3 {
		t.Errorf("Expected 3 allocated blocks for 35 tokens, got %d", len(table.BlockIDs))
	}
	if table.NumTokens != 35 {
		t.Errorf("Expected 35 tokens recorded, got %d", table.NumTokens)
	}

	// Free sequence and ensure blocks return to pool
	mgr.FreeSequence(seq1)
	if len(mgr.FreeBlocks) != 10 {
		t.Errorf("Expected 10 free blocks after release, got %d", len(mgr.FreeBlocks))
	}
}
