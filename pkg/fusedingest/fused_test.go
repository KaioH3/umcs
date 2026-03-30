package fusedingest

import (
	"testing"
)

func TestPlaceholder(t *testing.T) {
	// Fusedingest uses integration tests via the command
}

func TestMedianFloat64(t *testing.T) {
	tests := []struct {
		input    []float64
		expected float64
	}{
		{[]float64{1, 2, 3}, 2},
		{[]float64{1, 2, 3, 4}, 2.5},
		{[]float64{}, 0},
		{[]float64{5}, 5},
	}

	for _, tt := range tests {
		result := medianFloat64(tt.input)
		if result != tt.expected {
			t.Errorf("medianFloat64(%v) = %v, want %v", tt.input, result, tt.expected)
		}
	}
}

func TestVADToLevel(t *testing.T) {
	tests := []struct {
		input    float64
		expected string
	}{
		{-0.8, "LOW"},
		{0, "MED"},
		{0.8, "HIGH"},
	}

	for _, tt := range tests {
		result := vadToLevel(tt.input, -1.0, 1.0)
		if result != tt.expected {
			t.Errorf("vadToLevel(%v) = %v, want %v", tt.input, result, tt.expected)
		}
	}
}
