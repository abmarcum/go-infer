package quant

import (
	"math"
)

// FP16ToFP32 converts an IEEE 754 half-precision float (16-bit) to single-precision float32.
func FP16ToFP32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mant := uint32(h & 0x03FF)

	var f uint32
	if exp == 0 {
		if mant == 0 {
			f = sign
		} else {
			// Subnormal number
			exp = 1
			for (mant & 0x0400) == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x03FF
			f = sign | ((exp + (127 - 15)) << 23) | (mant << 13)
		}
	} else if exp == 0x1F {
		// Infinity or NaN
		f = sign | 0x7F800000 | (mant << 13)
	} else {
		// Normalized number
		f = sign | ((exp + (127 - 15)) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
}

// FP32ToFP16 converts a single-precision float32 to an IEEE 754 half-precision float (16-bit).
func FP32ToFP16(val float32) uint16 {
	bits := math.Float32bits(val)
	sign := (bits >> 16) & 0x8000
	exp := int((bits >> 23) & 0xFF)
	mant := bits & 0x007FFFFF

	if exp == 255 {
		if mant != 0 {
			// NaN
			return uint16(sign | 0x7E00)
		}
		// Infinity
		return uint16(sign | 0x7C00)
	}

	exp = exp - 127 + 15
	if exp >= 31 {
		// Overflow to Infinity
		return uint16(sign | 0x7C00)
	}
	if exp <= 0 {
		if exp < -10 {
			// Underflow to zero
			return uint16(sign)
		}
		// Subnormal
		mant = (mant | 0x00800000) >> (1 - exp)
		return uint16(sign | (mant >> 13))
	}

	return uint16(sign | (uint32(exp) << 10) | (mant >> 13))
}
