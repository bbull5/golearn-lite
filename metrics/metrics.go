package metrics

import (
	"math"
)


// Classification Metrics

func Accuracy(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0.0
	}

	correct := 0.0
	for i := range yTrue {
		if int(yTrue[i]) == int(yPred[i]) {
			correct++
		}
	}

	return correct / float64(len(yTrue))
}

func Precision(yTrue, yPred []float64) float64 {
	var tp, fp float64

	for i := range yTrue {
		if int(yPred[i]) == 1 {
			if int(yTrue[i]) == 1 {
				tp++
			} else {
				fp++
			}
		}
	}

	if tp+fp == 0 {
		return 0.0
	}
	return tp / (tp + fp)
}

func Recall(yTrue, yPred []float64) float64 {
	var tp, fn float64

	for i := range yTrue {
		if int(yTrue[i]) == 1 {
			if int(yPred[i]) == 1 {
				tp++
			} else {
				fn++
			}
		}
	}

	if tp+fn == 0 {
		return 0.0
	}
	return tp / (tp + fn)
}

func CrossEntropy(yTrue, yPred []float64) float64 {
	var loss float64

	n := len(yTrue)
	if n != len(yPred) || n == 0 {
		return 0.0
	}

	eps := 1e-15
	for i := 0; i < n; i++ {
		pred := math.Max(eps, math.Min(1-eps, yPred[i]))
		if int(yTrue[i]) == 1 {
			loss -= math.Log(pred)
		} else {
			loss -= math.Log(1 - pred)
		}
	}

	return loss / float64(n)
}

// Regression Metrics

func MAE(yTrue, yPred []float64) float64 {
	var sum float64
	n := len(yTrue)
	if n != len(yPred) || n == 0 {
		return 0.0
	}

	for i := 0; i < n; i++ {
		sum += math.Abs(yTrue[i] - yPred[i])
	}

	return sum / float64(n)
}

func MSE(yTrue, yPred []float64) float64 {
	var sum float64
	n := len(yTrue)
	if n != len(yPred) || n == 0 {
		return 0.0
	}

	for i := 0; i < n; i++ {
		diff := yTrue[i] - yPred[i]
		sum += diff * diff
	}

	return sum / float64(n)
}

func RMSE(yTrue, yPred []float64) float64 {
	return math.Sqrt(MSE(yTrue, yPred))
}

func R2(yTrue, yPred []float64) float64 {
	var ssRes, ssTot, mean float64
	n := len(yTrue)
	if n != len(yPred) || n == 0 {
		return 0.0
	}

	for _, val := range yTrue {
		mean += val
	}
	mean /= float64(n)

	for i := 0; i < n; i++ {
		diff := yTrue[i] - yPred[i]
		ssRes += diff * diff
		ssTot += (yTrue[i] - mean) * (yTrue[i] - mean)
	}

	if ssTot == 0 {
		return 0.0
	}
	return 1 - (ssRes / ssTot)
}
