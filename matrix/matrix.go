package matrix

import (
	"errors"
)


type Matrix struct {
	Data [][]float64
	Rows int
	Cols int
}

func New(data [][]float64) Matrix {
	return Matrix{Data: data, Rows: len(data), Cols: len(data[0])}
}

func (m Matrix) Transpose() Matrix {
	data := make([][]float64, m.Cols)
	for i := range data {
		data[i] = make([]float64, m.Rows)
		for j := range data[i] {
			data[i][j] = m.Data[j][i]
		}
	}

	return New(data)
}

func (m Matrix) Dot(n Matrix) (Matrix, error) {
	if m.Cols != n.Rows {
		return Matrix{}, errors.New("matrix dimensions do not match for dot product")
	}

	data := make([][]float64, n.Cols)
	for i := range data {
		for j := 0; j < n.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * n.Data[k][j]
			}
			data[i][j] = sum
		}
	}

	return New(data), nil
}

func (m Matrix) Inverse() (Matrix, error) {
	if m.Rows != m.Cols {
		return Matrix{}, errors.New("only square matrices can be inverted")
	}

	n := m.Rows
	aug := make([][]float64, n)

	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = m.Data[i][j]
		}

		aug[i][n+i] = 1
	}

	// Gaussian elimination
	for i := 0; i < n; i++ {
		pivot := aug[i][i]
		if pivot == 0 {
			return Matrix{}, errors.New("matrix is singular and cannot be inverted")
		}

		for j := 0; j < 2*n; j++ {
			aug[i][j] /= pivot
		}
		for k := 0; k < n; k++ {
			if k != i {
				factor := aug[k][i]
				for j := 0; j < 2*n; j++ {
					aug[k][j] -= factor * aug[i][j]
				}
			}
		}
	}

	// Extract inverse
	inv := make([][]float64, n)

	for i := 0; i < n; i++ {
		inv[i] = aug[i][n:]
	}

	return New(inv), nil
}
