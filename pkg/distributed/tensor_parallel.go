package distributed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// TPCoordinator coordinates Tensor Parallelism AllReduce operations across distributed workers.
type TPCoordinator struct {
	Rank       int
	WorldSize  int
	Peers      []string // URLs of all peer nodes (including self)
	HTTPClient *http.Client
	mu         sync.Mutex
	stepSeq    uint64
}

// NewTPCoordinator creates a new Tensor Parallelism coordinator for rank i out of N peers.
func NewTPCoordinator(rank int, peers []string) *TPCoordinator {
	return &TPCoordinator{
		Rank:       rank,
		WorldSize:  len(peers),
		Peers:      peers,
		HTTPClient: &http.Client{Timeout: 5 * time.Second},
	}
}

// AllReduce sums partial result vectors from all peer ranks and returns the synchronized total.
func (c *TPCoordinator) AllReduce(localVec []float32) ([]float32, error) {
	if c.WorldSize <= 1 {
		return localVec, nil
	}

	c.mu.Lock()
	c.stepSeq++
	step := c.stepSeq
	c.mu.Unlock()

	result := make([]float32, len(localVec))
	copy(result, localVec)

	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	// Broadcast local partial vector to all other peer ranks
	for i, peerURL := range c.Peers {
		if i == c.Rank {
			continue
		}

		wg.Add(1)
		go func(url string, peerRank int) {
			defer wg.Done()

			reqBody := TPReduceRequest{
				Rank:   c.Rank,
				StepID: step,
				Vector: localVec,
			}
			data, err := json.Marshal(reqBody)
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				errMu.Unlock()
				return
			}

			resp, err := c.HTTPClient.Post(url+"/v1/dist/tp-reduce", "application/json", bytes.NewReader(data))
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("peer %s error: %w", url, err)
				}
				errMu.Unlock()
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("peer %s returned HTTP %s", url, resp.Status)
				}
				errMu.Unlock()
				return
			}

			var res TPReduceResponse
			if err := json.NewDecoder(resp.Body).Decode(&res); err == nil && len(res.SumVector) == len(result) {
				errMu.Lock()
				for j := range result {
					result[j] += res.SumVector[j]
				}
				errMu.Unlock()
			}
		}(peerURL, i)
	}

	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}

	return result, nil
}
