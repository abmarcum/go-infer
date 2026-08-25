package math

import (
	"math"
)

// RMSNorm computes Root Mean Square Layer Normalization:
// out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
func RMSNorm(out, x, weight []float32, eps float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	mean := sum / float32(len(x))
	scale := float32(1.0 / math.Sqrt(float64(mean+eps)))
	for i := range out {
		out[i] = x[i] * scale * weight[i]
	}
}

// ApplyRoPE applies Rotary Position Embeddings to a vector slice (headDim elements)
func ApplyRoPE(vec []float32, pos, headDim int, theta float32) {
	half := headDim / 2
	for i := 0; i < half; i++ {
		freq := 1.0 / math.Pow(float64(theta), float64(2*i)/float64(headDim))
		val := float64(pos) * freq
		cosVal := float32(math.Cos(val))
		sinVal := float32(math.Sin(val))

		v0 := vec[i]
		v1 := vec[i+half]
		vec[i] = v0*cosVal - v1*sinVal
		vec[i+half] = v0*sinVal + v1*cosVal
	}
}

// SwiGLU computes in-place: gate[i] = SiLU(gate[i]) * up[i]
func SwiGLU(gate, up []float32, dim int) {
	for i := 0; i < dim; i++ {
		g := gate[i]
		// SiLU(x) = x / (1 + exp(-x))
		silu := g / (1.0 + float32(math.Exp(float64(-g))))
		gate[i] = silu * up[i]
	}
}

// Softmax computes in-place softmax over slice x
func Softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i := range x {
		val := float32(math.Exp(float64(x[i] - maxVal)))
		x[i] = val
		sum += val
	}
	if sum > 0 {
		invSum := 1.0 / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}
