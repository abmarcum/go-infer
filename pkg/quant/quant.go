package quant

import (
	"encoding/binary"
	"math"
	"unsafe"
)

// BlockQ4_0 represents a 32-weight Q4_0 quantized block (18 bytes total)
type BlockQ4_0 struct {
	D  uint16   // FP16 scale factor
	Qs [16]byte // 32 nibbles (4-bit signed ints, offset by 8)
}

// BlockQ4_1 represents a 32-weight Q4_1 quantized block (20 bytes total)
type BlockQ4_1 struct {
	D  uint16   // FP16 scale factor
	M  uint16   // FP16 min factor
	Qs [16]byte // 32 nibbles
}

// BlockQ5_0 represents a 32-weight Q5_0 quantized block (22 bytes total)
type BlockQ5_0 struct {
	D  uint16   // FP16 scale factor
	Qh [4]byte  // 5th bit for 32 weights
	Qs [16]byte // Lower 4 bits
}

// BlockQ5_1 represents a 32-weight Q5_1 quantized block (24 bytes total)
type BlockQ5_1 struct {
	D  uint16   // FP16 scale factor
	M  uint16   // FP16 min factor
	Qh [4]byte  // 5th bit for 32 weights
	Qs [16]byte // Lower 4 bits
}

// BlockQ8_0 represents a 32-weight Q8_0 quantized block (34 bytes total)
type BlockQ8_0 struct {
	D  uint16   // FP16 scale factor
	Qs [32]int8 // 32 int8 weights
}

// BlockQ2_K represents a 256-weight Q2_K quantized block (84 bytes total)
type BlockQ2_K struct {
	Scales [16]uint8 // 16 4-bit scales & mins packed into 16 bytes
	Qs     [64]uint8 // 256 2-bit weight values (4 weights per byte)
	D      uint16    // FP16 scale
	Dmin   uint16    // FP16 min
}

// BlockQ3_K represents a 256-weight Q3_K quantized block (110 bytes total)
type BlockQ3_K struct {
	Hmask  [32]uint8 // High 1-bit masks
	Qs     [64]uint8 // Low 2-bit values
	Scales [12]uint8 // 6-bit sub-block scales
	D      uint16    // FP16 master scale
}

// BF16ToFP32 converts a 16-bit Bfloat16 to float32
func BF16ToFP32(u uint16) float32 {
	return math.Float32frombits(uint32(u) << 16)
}

// DequantizeF32 converts raw little-endian bytes to []float32
func DequantizeF32(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	if len(data) < numElements*4 {
		return out
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), numElements)
	copy(out, src)
	return out
}

// DequantizeF16 converts raw FP16 bytes to []float32
func DequantizeF16(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	for i := 0; i < numElements && i*2+1 < len(data); i++ {
		h := binary.LittleEndian.Uint16(data[i*2:])
		out[i] = FP16ToFP32(h)
	}
	return out
}

// DequantizeBF16 converts raw BF16 bytes to []float32
func DequantizeBF16(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	for i := 0; i < numElements && i*2+1 < len(data); i++ {
		h := binary.LittleEndian.Uint16(data[i*2:])
		out[i] = BF16ToFP32(h)
	}
	return out
}

// DequantizeQ4_0 converts Q4_0 byte blocks to []float32
func DequantizeQ4_0(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 32
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+18 <= len(data); i++ {
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset:]))
		offset += 2
		for j := 0; j < 16; j++ {
			b := data[offset+j]
			x0 := int(b&0x0F) - 8
			x1 := int((b>>4)&0x0F) - 8
			out[outIdx+j] = float32(x0) * d
			out[outIdx+j+16] = float32(x1) * d
		}
		offset += 16
		outIdx += 32
	}
	return out
}

// DequantizeQ4_1 converts Q4_1 byte blocks to []float32
func DequantizeQ4_1(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 32
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+20 <= len(data); i++ {
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset:]))
		m := FP16ToFP32(binary.LittleEndian.Uint16(data[offset+2:]))
		offset += 4
		for j := 0; j < 16; j++ {
			b := data[offset+j]
			x0 := float32(b & 0x0F)
			x1 := float32((b >> 4) & 0x0F)
			out[outIdx+j] = x0*d + m
			out[outIdx+j+16] = x1*d + m
		}
		offset += 16
		outIdx += 32
	}
	return out
}

// DequantizeQ5_0 converts Q5_0 byte blocks to []float32
func DequantizeQ5_0(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 32
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+22 <= len(data); i++ {
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset:]))
		qh := binary.LittleEndian.Uint32(data[offset+2:])
		offset += 6
		for j := 0; j < 16; j++ {
			b := data[offset+j]
			h0 := (qh >> j) & 1
			h1 := (qh >> (j + 16)) & 1
			x0 := int(b&0x0F) | int(h0<<4) - 16
			x1 := int((b>>4)&0x0F) | int(h1<<4) - 16
			out[outIdx+j] = float32(x0) * d
			out[outIdx+j+16] = float32(x1) * d
		}
		offset += 16
		outIdx += 32
	}
	return out
}

// DequantizeQ5_1 converts Q5_1 byte blocks to []float32
func DequantizeQ5_1(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 32
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+24 <= len(data); i++ {
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset:]))
		m := FP16ToFP32(binary.LittleEndian.Uint16(data[offset+2:]))
		qh := binary.LittleEndian.Uint32(data[offset+4:])
		offset += 8
		for j := 0; j < 16; j++ {
			b := data[offset+j]
			h0 := (qh >> j) & 1
			h1 := (qh >> (j + 16)) & 1
			x0 := float32(int(b&0x0F) | int(h0<<4))
			x1 := float32(int((b>>4)&0x0F) | int(h1<<4))
			out[outIdx+j] = x0*d + m
			out[outIdx+j+16] = x1*d + m
		}
		offset += 16
		outIdx += 32
	}
	return out
}

// DequantizeQ8_0 converts Q8_0 byte blocks to []float32
func DequantizeQ8_0(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 32
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+34 <= len(data); i++ {
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset:]))
		offset += 2
		for j := 0; j < 32; j++ {
			out[outIdx+j] = float32(int8(data[offset+j])) * d
		}
		offset += 32
		outIdx += 32
	}
	return out
}

// DequantizeQ4_K converts Q4_K byte blocks (256 elements per 144 bytes) to []float32
func DequantizeQ4_K(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 256
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+144 <= len(data); i++ {
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset:]))
		dmin := FP16ToFP32(binary.LittleEndian.Uint16(data[offset+2:]))
		scales := data[offset+4 : offset+16]
		qs := data[offset+16 : offset+144]
		offset += 144

		// 8 sub-blocks of 32 elements
		for sb := 0; sb < 8; sb++ {
			var sc, m float32
			if sb < 4 {
				sc = float32(scales[sb]&63) * d
				m = float32(scales[sb+4]&63) * dmin
			} else {
				sc = float32((scales[sb+4]&0xF)|((scales[sb-4]>>6)<<4)) * d
				m = float32((scales[sb+4]>>4)|((scales[sb]>>6)<<4)) * dmin
			}

			qOffset := sb * 16
			for j := 0; j < 16; j++ {
				b := qs[qOffset+j]
				x0 := float32(b & 0x0F)
				x1 := float32((b >> 4) & 0x0F)
				out[outIdx+sb*32+j] = x0*sc - m
				out[outIdx+sb*32+j+16] = x1*sc - m
			}
		}
		outIdx += 256
	}
	return out
}

// DequantizeQ6_K converts Q6_K byte blocks (256 elements per 210 bytes) to []float32
func DequantizeQ6_K(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 256
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+210 <= len(data); i++ {
		ql := data[offset : offset+128]
		qh := data[offset+128 : offset+192]
		scales := data[offset+192 : offset+208]
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset+208:]))
		offset += 210

		for sb := 0; sb < 16; sb++ {
			sc := float32(int8(scales[sb])) * d
			for j := 0; j < 16; j++ {
				idx := sb*16 + j
				l := ql[idx/2]
				var qVal int
				if idx%2 == 0 {
					qVal = int(l & 0x0F)
				} else {
					qVal = int((l >> 4) & 0x0F)
				}
				h := (qh[idx/4] >> ((idx % 4) * 2)) & 3
				qVal = (qVal | (int(h) << 4)) - 32
				out[outIdx+idx] = float32(qVal) * sc
			}
		}
		outIdx += 256
	}
	return out
}

// DotVecQ4_0 performs a direct dot product between activation vector x (F32) and quantized weights Q4_0
func DotVecQ4_0(x []float32, rawBlocks []byte, numElements int) float32 {
	numBlocks := numElements / 32
	var total float32

	blockPtr := (*BlockQ4_0)(unsafe.Pointer(&rawBlocks[0]))
	blocks := unsafe.Slice(blockPtr, numBlocks)

	for b := 0; b < numBlocks; b++ {
		blk := &blocks[b]
		d := FP16ToFP32(blk.D)
		xOffset := b * 32

		var blockSum float32
		for j := 0; j < 16; j++ {
			byteVal := blk.Qs[j]
			v0 := float32(int(byteVal&0x0F) - 8)
			v1 := float32(int((byteVal>>4)&0x0F) - 8)

			blockSum += v0*x[xOffset+j] + v1*x[xOffset+j+16]
		}
		total += blockSum * d
	}
	return total
}

// DotVecQ8_0 performs a direct dot product between activation vector x (F32) and quantized weights Q8_0
func DotVecQ8_0(x []float32, rawBlocks []byte, numElements int) float32 {
	numBlocks := numElements / 32
	var total float32

	blockPtr := (*BlockQ8_0)(unsafe.Pointer(&rawBlocks[0]))
	blocks := unsafe.Slice(blockPtr, numBlocks)

	for b := 0; b < numBlocks; b++ {
		blk := &blocks[b]
		d := FP16ToFP32(blk.D)
		xOffset := b * 32

		var blockSum float32
		for j := 0; j < 32; j++ {
			blockSum += float32(blk.Qs[j]) * x[xOffset+j]
		}
		total += blockSum * d
	}
	return total
}

// DequantizeQ2_K converts Q2_K byte blocks (256 elements per 84 bytes) to []float32
func DequantizeQ2_K(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 256
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+84 <= len(data); i++ {
		scales := data[offset : offset+16]
		qs := data[offset+16 : offset+80]
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset+80:]))
		dmin := FP16ToFP32(binary.LittleEndian.Uint16(data[offset+82:]))
		offset += 84

		for sb := 0; sb < 16; sb++ {
			sc := float32(scales[sb]&0x0F) * d
			m := float32(scales[sb]>>4) * dmin
			for j := 0; j < 16; j++ {
				idx := sb*16 + j
				byteIdx := idx / 4
				shift := (idx % 4) * 2
				q := float32((qs[byteIdx] >> shift) & 3)
				out[outIdx+idx] = q*sc - m
			}
		}
		outIdx += 256
	}
	return out
}

// DequantizeQ3_K converts Q3_K byte blocks (256 elements per 110 bytes) to []float32
func DequantizeQ3_K(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	numBlocks := numElements / 256
	offset := 0
	outIdx := 0

	for i := 0; i < numBlocks && offset+110 <= len(data); i++ {
		hmask := data[offset : offset+32]
		qs := data[offset+32 : offset+96]
		scales := data[offset+96 : offset+108]
		d := FP16ToFP32(binary.LittleEndian.Uint16(data[offset+108:]))
		offset += 110

		for sb := 0; sb < 16; sb++ {
			sc := float32(int8(scales[sb%12])) * d
			for j := 0; j < 16; j++ {
				idx := sb*16 + j
				byteIdx := idx / 4
				shift := (idx % 4) * 2
				low2 := (qs[byteIdx] >> shift) & 3

				hByte := idx / 8
				hShift := idx % 8
				high1 := (hmask[hByte] >> hShift) & 1

				q := int(low2) | (int(high1) << 2) - 4
				out[outIdx+idx] = float32(q) * sc
			}
		}
		outIdx += 256
	}
	return out
}

// QuantizeQ8_0 quantizes a float32 slice into Q8_0 blocks
func QuantizeQ8_0(src []float32) []byte {
	numBlocks := (len(src) + 31) / 32
	out := make([]byte, numBlocks*34)

	for b := 0; b < numBlocks; b++ {
		start := b * 32
		end := start + 32
		if end > len(src) {
			end = len(src)
		}

		var maxVal float32
		for i := start; i < end; i++ {
			v := float32(math.Abs(float64(src[i])))
			if v > maxVal {
				maxVal = v
			}
		}

		scale := maxVal / 127.0
		var scaleFP16 uint16
		if scale > 0 {
			scaleFP16 = FP32ToFP16(scale)
			scale = FP16ToFP32(scaleFP16) // Quantize scale
		}
		invScale := float32(0)
		if scale > 0 {
			invScale = 1.0 / scale
		}

		offset := b * 34
		binary.LittleEndian.PutUint16(out[offset:offset+2], scaleFP16)

		for i := 0; i < 32; i++ {
			var qVal int8
			if start+i < len(src) && invScale > 0 {
				v := src[start+i] * invScale
				if v > 127 {
					v = 127
				} else if v < -128 {
					v = -128
				}
				qVal = int8(math.Round(float64(v)))
			}
			out[offset+2+i] = byte(qVal)
		}
	}
	return out
}

// QuantizeQ4_0 quantizes a float32 slice into Q4_0 blocks
func QuantizeQ4_0(src []float32) []byte {
	numBlocks := (len(src) + 31) / 32
	out := make([]byte, numBlocks*18)

	for b := 0; b < numBlocks; b++ {
		start := b * 32
		end := start + 32
		if end > len(src) {
			end = len(src)
		}

		var maxVal float32
		for i := start; i < end; i++ {
			v := float32(math.Abs(float64(src[i])))
			if v > maxVal {
				maxVal = v
			}
		}

		scale := maxVal / 7.0
		var scaleFP16 uint16
		if scale > 0 {
			scaleFP16 = FP32ToFP16(scale)
			scale = FP16ToFP32(scaleFP16)
		}
		invScale := float32(0)
		if scale > 0 {
			invScale = 1.0 / scale
		}

		offset := b * 18
		binary.LittleEndian.PutUint16(out[offset:offset+2], scaleFP16)

		for j := 0; j < 16; j++ {
			var q0, q1 int
			if start+j < len(src) && invScale > 0 {
				v := src[start+j] * invScale
				q0 = int(math.Round(float64(v))) + 8
				if q0 < 0 {
					q0 = 0
				} else if q0 > 15 {
					q0 = 15
				}
			} else {
				q0 = 8
			}

			if start+j+16 < len(src) && invScale > 0 {
				v := src[start+j+16] * invScale
				q1 = int(math.Round(float64(v))) + 8
				if q1 < 0 {
					q1 = 0
				} else if q1 > 15 {
					q1 = 15
				}
			} else {
				q1 = 8
			}

			out[offset+2+j] = byte(q0 | (q1 << 4))
		}
	}
	return out
}

// DotVecQ2_K performs dot product for Q2_K block
func DotVecQ2_K(x []float32, rawBlocks []byte, numElements int) float32 {
	deq := DequantizeQ2_K(rawBlocks, numElements)
	return DotVecF32(x, deq)
}

// DotVecQ3_K performs dot product for Q3_K block
func DotVecQ3_K(x []float32, rawBlocks []byte, numElements int) float32 {
	deq := DequantizeQ3_K(rawBlocks, numElements)
	return DotVecF32(x, deq)
}

// DotVecQ4_K performs a dot product for Q4_K block
func DotVecQ4_K(x []float32, rawBlocks []byte, numElements int) float32 {
	deq := DequantizeQ4_K(rawBlocks, numElements)
	return DotVecF32(x, deq)
}

// DotVecQ6_K performs a dot product for Q6_K block
func DotVecQ6_K(x []float32, rawBlocks []byte, numElements int) float32 {
	deq := DequantizeQ6_K(rawBlocks, numElements)
	return DotVecF32(x, deq)
}

// DotVecF32 computes vector dot product of two float32 slices
func DotVecF32(x, w []float32) float32 {
	n := len(x)
	var sum float32
	i := 0
	for ; i <= n-8; i += 8 {
		sum += x[i]*w[i] +
			x[i+1]*w[i+1] +
			x[i+2]*w[i+2] +
			x[i+3]*w[i+3] +
			x[i+4]*w[i+4] +
			x[i+5]*w[i+5] +
			x[i+6]*w[i+6] +
			x[i+7]*w[i+7]
	}
	for ; i < n; i++ {
		sum += x[i] * w[i]
	}
	return sum
}

// DotVecF16 computes dot product between F32 vector x and raw FP16 weights
func DotVecF16(x []float32, rawF16 []byte, numElements int) float32 {
	var sum float32
	for i := 0; i < numElements && i*2+1 < len(rawF16); i++ {
		h := binary.LittleEndian.Uint16(rawF16[i*2:])
		w := FP16ToFP32(h)
		sum += x[i] * w
	}
	return sum
}

// DotVecBF16 computes dot product between F32 vector x and raw BF16 weights
func DotVecBF16(x []float32, rawBF16 []byte, numElements int) float32 {
	var sum float32
	for i := 0; i < numElements && i*2+1 < len(rawBF16); i++ {
		h := binary.LittleEndian.Uint16(rawBF16[i*2:])
		w := BF16ToFP32(h)
		sum += x[i] * w
	}
	return sum
}
