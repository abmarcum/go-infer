package sampler

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

// Params holds sampling hyperparameters.
type Params struct {
	Temperature float32
	TopP        float32
	TopK        int
	RepPenalty  float32
	Rand        *rand.Rand
}

// DefaultParams returns balanced standard sampling configuration.
func DefaultParams() Params {
	return Params{
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        40,
		RepPenalty:  1.1,
		Rand:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SampleToken selects the next token from logits according to sampling parameters and history.
func SampleToken(logits []float32, history []int, params Params) int {
	if len(logits) == 0 {
		return 0
	}

	// 1. Repetition Penalty
	if params.RepPenalty != 1.0 && params.RepPenalty > 0 && len(history) > 0 {
		for _, tok := range history {
			if tok >= 0 && tok < len(logits) {
				if logits[tok] < 0 {
					logits[tok] *= params.RepPenalty
				} else {
					logits[tok] /= params.RepPenalty
				}
			}
		}
	}

	// 2. Greedy Sampling (Temperature <= 0)
	if params.Temperature <= 0.0 {
		best := 0
		maxLogit := logits[0]
		for i, l := range logits {
			if l > maxLogit {
				maxLogit = l
				best = i
			}
		}
		return best
	}

	// 3. Temperature Scaling
	invTemp := float32(1.0) / params.Temperature

	// Find top candidates without allocating/sorting 152k elements
	type probPair struct {
		id    int
		logit float32
	}

	k := params.TopK
	if k <= 0 || k > len(logits) {
		k = 40
	}
	if k > len(logits) {
		k = len(logits)
	}

	// Maintain top-k elements using simple linear insertion / bounded array
	topList := make([]probPair, 0, k)
	for i, l := range logits {
		scaled := l * invTemp
		if len(topList) < k {
			topList = append(topList, probPair{id: i, logit: scaled})
			if len(topList) == k {
				sort.Slice(topList, func(a, b int) bool {
					return topList[a].logit > topList[b].logit
				})
			}
		} else if scaled > topList[k-1].logit {
			// Insert in sorted position
			pos := k - 1
			for pos > 0 && topList[pos-1].logit < scaled {
				topList[pos] = topList[pos-1]
				pos--
			}
			topList[pos] = probPair{id: i, logit: scaled}
		}
	}

	if len(topList) == 0 {
		return 0
	}

	// Softmax over top-k candidates
	maxLogit := topList[0].logit
	var sumExp float32
	probs := make([]float32, len(topList))
	for i, p := range topList {
		expVal := math_exp(p.logit - maxLogit)
		probs[i] = expVal
		sumExp += expVal
	}
	for i := range probs {
		probs[i] /= sumExp
	}

	// 5. Top-P (Nucleus) Truncation
	cutoffIdx := len(topList)
	if params.TopP > 0 && params.TopP < 1.0 {
		var cumSum float32
		for i, pr := range probs {
			cumSum += pr
			if cumSum >= params.TopP {
				cutoffIdx = i + 1
				break
			}
		}
	}
	topList = topList[:cutoffIdx]
	probs = probs[:cutoffIdx]

	// 6. Cumulative Probability Sampling
	var normSum float32
	for _, pr := range probs {
		normSum += pr
	}

	rng := params.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	r := rng.Float32() * normSum
	var acc float32
	for i, pr := range probs {
		acc += pr
		if r <= acc {
			return topList[i].id
		}
	}

	return topList[0].id
}

func math_exp(x float32) float32 {
	if x < -88.0 {
		return 0.0
	}
	if x > 88.0 {
		return 1.6516363e+38
	}
	return float32(math.Exp(float64(x)))
}
