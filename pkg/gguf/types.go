package gguf

const (
	Magic = 0x46554747 // "GGUF" in Little Endian

	Version2 = 2
	Version3 = 3

	// GGUF Metadata Value Types
	TypeUint8   = 0
	TypeInt8    = 1
	TypeUint16  = 2
	TypeInt16   = 3
	TypeUint32  = 4
	TypeInt32   = 5
	TypeFloat32 = 6
	TypeBool    = 7
	TypeString  = 8
	TypeArray   = 9
	TypeUint64  = 10
	TypeInt64   = 11
	TypeFloat64 = 12

	// GGML Quantization Types
	GGMLTypeF32     = 0
	GGMLTypeF16     = 1
	GGMLTypeQ4_0    = 2
	GGMLTypeQ4_1    = 3
	GGMLTypeQ5_0    = 6
	GGMLTypeQ5_1    = 7
	GGMLTypeQ8_0    = 8
	GGMLTypeQ8_1    = 9
	GGMLTypeQ2_K    = 10
	GGMLTypeQ3_K    = 11
	GGMLTypeQ4_K    = 12
	GGMLTypeQ5_K    = 13
	GGMLTypeQ6_K    = 14
	GGMLTypeQ8_K    = 15
	GGMLTypeIQ2_XXS = 16
	GGMLTypeIQ2_XS  = 17
	GGMLTypeIQ3_XXS = 18
	GGMLTypeIQ1_S   = 19
	GGMLTypeIQ4_NL  = 20
	GGMLTypeIQ3_S   = 21
	GGMLTypeIQ2_S   = 22
	GGMLTypeIQ4_XS  = 23
	GGMLTypeI8      = 24
	GGMLTypeI16     = 25
	GGMLTypeI32     = 26
	GGMLTypeI64     = 27
	GGMLTypeF64     = 28
	GGMLTypeBF16    = 29
)

// TensorInfo holds tensor metadata from the GGUF header
type TensorInfo struct {
	Name       string
	Dimensions []uint64 // Dimensions in GGML order (dim 0 is columns / innermost)
	Type       uint32
	Offset     uint64
}

// NumElements returns total number of elements in the tensor
func (t *TensorInfo) NumElements() int {
	if len(t.Dimensions) == 0 {
		return 0
	}
	n := 1
	for _, d := range t.Dimensions {
		n *= int(d)
	}
	return n
}

// Header represents parsed GGUF header and metadata
type Header struct {
	Version     uint32
	TensorCount uint64
	KVCount     uint64
	Metadata    map[string]interface{}
	Tensors     map[string]TensorInfo
	DataOffset  int64
}
