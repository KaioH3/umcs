package analyze

import (
	"bytes"
	"encoding/binary"
	"strconv"
	"strings"
)

type CompressionAlgorithm string

const (
	AlgoRLE   CompressionAlgorithm = "rle"
	AlgoLZ77  CompressionAlgorithm = "lz77"
	AlgoDelta CompressionAlgorithm = "delta"
	AlgoBWT   CompressionAlgorithm = "bwt"
)

type CompressionResult struct {
	OriginalSize   int                  `json:"original_size"`
	Compressed     interface{}          `json:"compressed"`
	CompressedSize int                  `json:"compressed_size"`
	Ratio          float64              `json:"ratio"`
	Algorithm      CompressionAlgorithm `json:"algorithm"`
}

func Compress(data interface{}, algo CompressionAlgorithm) *CompressionResult {
	switch v := data.(type) {
	case string:
		return compressString(v, algo)
	case []int:
		return compressInts(v, algo)
	case []int32:
		return compressInt32s(v, algo)
	default:
		return &CompressionResult{
			OriginalSize:   0,
			Compressed:     "unsupported type",
			CompressedSize: 0,
			Ratio:          1.0,
			Algorithm:      algo,
		}
	}
}

func compressString(data string, algo CompressionAlgorithm) *CompressionResult {
	var compressed string
	var originalSize = len(data)

	switch algo {
	case AlgoRLE:
		compressed = rleEncode(data)
	case AlgoLZ77:
		compressed = lz77Encode(data)
	case AlgoBWT:
		compressed = bwtEncode(data)
	default:
		compressed = data
	}

	compressedSize := len(compressed)
	ratio := float64(compressedSize) / float64(originalSize)
	if originalSize == 0 {
		ratio = 1.0
	}

	return &CompressionResult{
		OriginalSize:   originalSize,
		Compressed:     compressed,
		CompressedSize: compressedSize,
		Ratio:          ratio,
		Algorithm:      algo,
	}
}

func compressInts(data []int, algo CompressionAlgorithm) *CompressionResult {
	var compressed interface{}
	originalSize := len(data)

	switch algo {
	case AlgoDelta:
		compressed = deltaEncode(data)
	case AlgoRLE:
		compressed = rleEncodeInts(data)
	default:
		compressed = data
	}

	var compressedSize int
	switch c := compressed.(type) {
	case []int:
		compressedSize = len(c) * 4
	case string:
		compressedSize = len(c)
	default:
		compressedSize = originalSize * 4
	}

	ratio := float64(compressedSize) / float64(originalSize*4)
	if originalSize == 0 {
		ratio = 1.0
	}

	return &CompressionResult{
		OriginalSize:   originalSize * 4,
		Compressed:     compressed,
		CompressedSize: compressedSize,
		Ratio:          ratio,
		Algorithm:      algo,
	}
}

func compressInt32s(data []int32, algo CompressionAlgorithm) *CompressionResult {
	var compressed interface{}
	originalSize := len(data)

	switch algo {
	case AlgoDelta:
		compressed = deltaEncodeInt32(data)
	default:
		compressed = data
	}

	var compressedSize int
	switch c := compressed.(type) {
	case []int32:
		compressedSize = len(c) * 4
	default:
		compressedSize = originalSize * 4
	}

	ratio := float64(compressedSize) / float64(originalSize*4)
	if originalSize == 0 {
		ratio = 1.0
	}

	return &CompressionResult{
		OriginalSize:   originalSize * 4,
		Compressed:     compressed,
		CompressedSize: compressedSize,
		Ratio:          ratio,
		Algorithm:      algo,
	}
}

func rleEncode(data string) string {
	if len(data) == 0 {
		return ""
	}

	var result strings.Builder
	runes := []rune(data)
	length := len(runes)

	i := 0
	for i < length {
		count := 1
		for i+count < length && runes[i+count] == runes[i] && count < 255 {
			count++
		}
		result.WriteString(strconv.Itoa(count))
		result.WriteRune(runes[i])
		i += count
	}

	return result.String()
}

func rleDecode(data string) string {
	if len(data) == 0 {
		return ""
	}

	var result strings.Builder
	length := len(data)
	i := 0

	for i < length {
		countStr := ""
		for i < length && data[i] >= '0' && data[i] <= '9' {
			countStr += string(data[i])
			i++
		}

		if i >= length {
			break
		}

		count, _ := strconv.Atoi(countStr)
		char := rune(data[i])

		for j := 0; j < count; j++ {
			result.WriteRune(char)
		}
		i++
	}

	return result.String()
}

func rleEncodeInts(data []int) []int {
	if len(data) == 0 {
		return []int{}
	}

	var result []int
	length := len(data)

	i := 0
	for i < length {
		count := 1
		for i+count < length && data[i+count] == data[i] && count < 255 {
			count++
		}
		result = append(result, count, data[i])
		i += count
	}

	return result
}

func lz77Encode(data string) string {
	if len(data) == 0 {
		return ""
	}

	const windowSize = 4096
	const maxLength = 255
	const minMatch = 3

	var result strings.Builder
	length := len(data)

	i := 0
	for i < length {
		bestOffset := 0
		bestLength := 0

		start := i - windowSize
		if start < 0 {
			start = 0
		}

		for j := start; j < i; j++ {
			matchLen := 0
			for i+matchLen < length && j+matchLen < i && data[j+matchLen] == data[i+matchLen] {
				matchLen++
				if matchLen >= maxLength || i+matchLen >= length {
					break
				}
			}

			if matchLen > bestLength {
				bestLength = matchLen
				bestOffset = i - j
			}
		}

		if bestLength >= minMatch {
			result.WriteString(strconv.Itoa(bestOffset))
			result.WriteRune(',')
			result.WriteString(strconv.Itoa(bestLength))
			result.WriteRune(',')
			i += bestLength
		} else {
			result.WriteString(strconv.Itoa(int(data[i])))
			result.WriteRune(';')
			i++
		}
	}

	return result.String()
}

func lz77Decode(data string) string {
	if len(data) == 0 {
		return ""
	}

	var result bytes.Buffer
	length := len(data)
	i := 0

	for i < length {
		hasComma := false
		for j := i; j < length; j++ {
			if data[j] == ',' {
				hasComma = true
				break
			}
			if data[j] == ';' {
				break
			}
		}

		if hasComma {
			offsetStr := ""
			for i < length && data[i] != ',' {
				offsetStr += string(data[i])
				i++
			}
			if i >= length {
				break
			}
			i++

			lenStr := ""
			for i < length && data[i] != ',' {
				lenStr += string(data[i])
				i++
			}

			offset, _ := strconv.Atoi(offsetStr)
			length2, _ := strconv.Atoi(lenStr)

			start := result.Len() - offset
			bytes := result.Bytes()
			for j := 0; j < length2; j++ {
				result.WriteByte(bytes[start+j])
			}
		} else {
			if data[i] == ';' {
				i++
				continue
			}
			numStr := ""
			for i < length && data[i] != ';' {
				numStr += string(data[i])
				i++
			}
			if numStr != "" {
				val, _ := strconv.Atoi(numStr)
				if val < 256 {
					result.WriteByte(byte(val))
				}
			}
			if i < length && data[i] == ';' {
				i++
			}
		}
	}

	return result.String()
}

func deltaEncode(data []int) []int {
	if len(data) == 0 {
		return []int{}
	}

	result := make([]int, len(data))
	prev := 0

	for i, val := range data {
		result[i] = val - prev
		prev = val
	}

	return result
}

func deltaDecode(data []int) []int {
	if len(data) == 0 {
		return []int{}
	}

	result := make([]int, len(data))
	prev := 0

	for i, val := range data {
		result[i] = val + prev
		prev = result[i]
	}

	return result
}

func deltaEncodeInt32(data []int32) []int32 {
	if len(data) == 0 {
		return []int32{}
	}

	result := make([]int32, len(data))
	var prev int32

	for i, val := range data {
		result[i] = val - prev
		prev = val
	}

	return result
}

func deltaDecodeInt32(data []int32) []int32 {
	if len(data) == 0 {
		return []int32{}
	}

	result := make([]int32, len(data))
	var prev int32

	for i, val := range data {
		result[i] = val + prev
		prev = result[i]
	}

	return result
}

func bwtEncode(data string) string {
	if len(data) <= 1 {
		return data
	}

	n := len(data)
	rotations := make([]string, n)

	for i := 0; i < n; i++ {
		rotations[i] = data[i:] + data[:i]
	}

	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if rotations[i] > rotations[j] {
				rotations[i], rotations[j] = rotations[j], rotations[i]
			}
		}
	}

	var result strings.Builder
	for i := 0; i < n; i++ {
		result.WriteByte(rotations[i][n-1])
	}

	return result.String()
}

func bwtDecode(data string) string {
	if len(data) <= 1 {
		return data
	}

	// Simplified BWT decode - for real BWT we'd need the full algorithm
	// This is a placeholder that reverses the basic transformation
	return data
}

func CompressBinary(data []byte, algo CompressionAlgorithm) ([]byte, error) {
	switch algo {
	case AlgoRLE:
		return rleEncodeBinary(data), nil
	default:
		return data, nil
	}
}

func rleEncodeBinary(data []byte) []byte {
	if len(data) == 0 {
		return []byte{}
	}

	var result bytes.Buffer
	length := len(data)
	i := 0

	for i < length {
		count := 1
		for i+count < length && data[i+count] == data[i] && count < 255 {
			count++
		}
		result.WriteByte(byte(count))
		result.WriteByte(data[i])
		i += count
	}

	return result.Bytes()
}

func DecompressBinary(data []byte, algo CompressionAlgorithm) ([]byte, error) {
	switch algo {
	case AlgoRLE:
		return rleDecodeBinary(data), nil
	default:
		return data, nil
	}
}

func rleDecodeBinary(data []byte) []byte {
	if len(data) == 0 || len(data)%2 != 0 {
		return []byte{}
	}

	var result bytes.Buffer
	length := len(data)

	for i := 0; i < length; i += 2 {
		count := int(data[i])
		char := data[i+1]
		for j := 0; j < count; j++ {
			result.WriteByte(char)
		}
	}

	return result.Bytes()
}

func IntToBytes(data []int) []byte {
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, data)
	return buf.Bytes()
}

func BytesToInt(data []byte) []int {
	var result []int
	buf := bytes.NewReader(data)
	binary.Read(buf, binary.LittleEndian, &result)
	return result
}
