package data

import (
	"os"
	"strconv"
	"errors"
	"math"
	"math/rand"
	"time"
	"sort"
	"encoding/csv"

	"golearn-lite/matrix"
)

func TrainTestSplit(X [][]float64, y []float64, testSize float64) (XTrain, XTest [][]float64, yTrain, yTest []float64, err error) {
	if len(X) != len(y) {
		return nil, nil, nil, nil, errors.New(" X and y must have the same number of samples")
	}
	if testSize <= 0.0 || testSize >= 1.0 {
		return nil, nil, nil, nil, errors.New("testSize must be between 0 and 1")
	}

	n := len(X)
	nTest := int(float64(n) * testSize)

	// Generate a list of indices and shuffle them
	indices := rand.Perm(n)
	rand.Seed(time.Now().UnixNano())

	// Create slices for test and train sets
	for i, idx := range indices {
		if i < nTest {
			XTest = append(XTest, X[idx])
			yTest = append(yTest, y[idx])
		} else {
			XTrain = append(XTrain, X[idx])
			yTrain = append(yTrain, y[idx])
		}
	}

	return XTrain, XTest, yTrain, yTest, nil
}

func Shuffle(X [][]float64, y []float64) ([][]float64, []float64, error) {
	if len(X) != len(y) {
		return nil, nil, errors.New(" X and y must have the same number of samples")
	}

	n := len(X)
	rand.Seed(time.Now().UnixNano())
	perm := rand.Perm(n)

	XShuffled := make([][]float64, n)
	yShuffled := make([]float64, n)

	for i, idx := range perm {
		XShuffled[i] = X[idx]
		yShuffled[i] = y[idx]
	}
	
	return XShuffled, yShuffled, nil
}

func Normalize(X [][]float64) [][]float64 {
	rows, cols := len(X), len(X[0])
	min := make([]float64, cols)
	max := make([]float64, cols)

	// Iniitialize min/max
	for j := 0; j < cols; j++ {
		min[j], max[j] = X[0][j], X[0][j]
	}

	// Find min/max
	for i := 1; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if X[i][j] < min[j] {
				min[j] = X[i][j]
			}
			if X[i][j] > max[j] {
				max[j] = X[i][j]
			}
		}
	}

	// Scale
	normalized := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		normalized[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			if max[j] == min[j] {
				normalized[i][j] = 0.0
			} else {
				normalized[i][j] = (X[i][j] - min[j]) / (max[j] - min[j])
			}
		}
	}

	return normalized
}

func Standardize(X [][]float64) [][]float64 {
	rows, cols := len(X), len(X[0])
	mean := make([]float64, cols)
	std := make([]float64, cols)

	// Compute mean
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			diff := X[i][j] - mean[j]
			std[j] += diff * diff
		}
		std[j] = math.Sqrt(std[j] / float64(rows))
	}

	// Apply z score scaling
	standardized := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		standardized[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			if std[j] == 0 {
				standardized[i][j] = 0.0
			} else {
				standardized[i][j] = (X[i][j] - mean[j]) / std[j]
			}
		}
	}

	return standardized
}

func OneHotEncode(y []int) [][]float64 {
	n := len(y)
	maxClass := 0

	for _, val := range y {
		if val > maxClass {
			maxClass = val
		}
	}
	
	numClasses := maxClass + 1
	encoded := make([][]float64, n)

	for i, label := range y {
		row := make([]float64, numClasses)
		row[label] = 1.0
		encoded[i] = row
	}

	return encoded
}

func ToMatrix(X [][]float64) matrix.Matrix {
	return matrix.New(X)
}

func FromMatrix(m matrix.Matrix) [][]float64 {
	return m.Data
}

func DropNaN(X [][]float64, y []float64) ([][]float64, []float64) {
	cleanX := make([][]float64, 0, len(X))
	cleany := make([]float64, 0, len(y))

	for i := range X {
		hasNaN := false
		for _, val := range X[i] {
			if math.IsNaN(val) {
				hasNaN = true
				break
			}
		}

		if !hasNaN {
			cleanX = append(cleanX, X[i])
			cleany = append(cleany, y[i])
		}
	}

	return cleanX, cleany
}

func ImputeNaN(X [][]float64, strategy string) [][]float64 {
	rows, cols := len(X), len(X[0])
	imputed := make([][]float64, rows)

	stats := make([]float64, cols)
	for j := 0; j < cols; j++ {
		col := []float64{}
		for i := 0; i < rows; i++ {
			val := X[i][j]
			if !math.IsNaN(val) {
				col = append(col, val)
			}
		}

		if len(col) == 0 {
			stats[j] = 0
			continue
		}

		switch strategy {
		case "mean":
			sum := 0.0
			for _, v := range col {
				sum += v
			}
			stats[j] = sum / float64(len(col))

		case "median":
			sort.Float64s(col)
			mid := len(col) / 2

			if len(col) % 2 == 0 {
				stats[j] = (col[mid-1] + col[mid]) / 2
			} else {
				stats[j] = col[mid]
			}
		default:
			stats[j] = 0.0
		}
	}

	// Apply imputation
	for i := 0; i < rows; i++ {
		row := make([]float64, cols)
		for j := 0; j < cols; j++ {
			if math.IsNaN(X[i][j]) {
				row[j] = stats[j]
			} else {
				row[j] = X[i][j]
			}
		}
		imputed[i] = row
	}

	return imputed
}

func LoadCSV(path string, hasHeader bool) ([][]float64, []float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if hasHeader {
		rows = rows[1:]
	}

	X := make([][]float64, 0, len(rows))
	y := make([]float64, 0, len(rows))

	for _, row := range rows {
		n := len(row)
		if n < 2 {
			return nil, nil, errors.New("CSV row must have at least 2 columns")
		}

		xrow := make([]float64, n-1)
		for i := 0; i < n; i++ {
			val, err := strconv.ParseFloat(row[i], 64)
			if err != nil {
				return nil, nil, err
			}
			xrow[i] = val
		}

		target, err := strconv.ParseFloat(row[n-1], 64)
		if err != nil {
			return nil, nil, err
		}

		X = append(X, xrow)
		y = append(y, target)
	}

	return X, y, nil
}
