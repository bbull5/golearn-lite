package naivebayes

import (
	"os"
	"errors"
	"math"
	"sort"
	"encoding/gob"

	"golearn-lite/core"
	"golearn-lite/matrix"
)


type GaussianNB struct {
	ClassPriors			map[float64]float64
	Means				map[float64][]float64
	Variances			map[float64][]float64
	Classes				[]float64
	Epsilon				float64
}


func NewGaussianNB() *GaussianNB {
	return &GaussianNB{
		ClassPriors: 	make(map[float64]float64),
		Means:			make(map[float64][]float64),
		Variances:		make(map[float64][]float64),
		Epsilon:		1e-9,
	}
}

func (gnb *GaussianNB) Fit(X matrix.Matrix, y []float64) error {
	if X.Rows != len(y) {
		return errors.New("number of samples in X and y must match")
	}

	nSamples, nFeatures := X.Rows, X.Cols
	classCounts := make(map[float64]int)

	// Get class labels
	classSet := make(map[float64]bool)

	for _, label := range y {
		classSet[label] = true
	}

	for label := range classSet {
		gnb.Classes = append(gnb.Classes, label)
	}

	// compute means and variance
	for _, class := range gnb.Classes {
		classCounts[class] = 0
		means := make([]float64, nFeatures)
		vars := make([]float64, nFeatures)

		// First pass: compute sums
		for i, row := range X.Data {
			if y[i] == class {
				classCounts[class]++
				for j := 0; j < nFeatures; j++ {
					means[j] += row[j]
				}
			}
		}

		for j := 0; j < nFeatures; j++ {
			means[j] /= float64(classCounts[class])
		}

		for i, row := range X.Data {
			if y[i] == class {
				for j := 0; j < nFeatures; j++ {
					diff := row[j] - means[j]
					vars[j] += diff * diff
				}
			}
		}

		for j := 0; j < nFeatures; j++ {
			vars[j] /= float64(classCounts[class])
		}

		gnb.Means[class] = means
		gnb.Variances[class] = vars
	}

	// compute priors
	for class, count := range classCounts {
		gnb.ClassPriors[class] = float64(count) / float64(nSamples)
	}

	return nil
}

func (gnb *GaussianNB) Predict(X matrix.Matrix) []float64 {
	n := X.Rows
	preds := make([]float64, n)

	for i := 0; i < n; i++ {
		x := X.Data[i]
		scores := make(map[float64]float64)

		for _, class := range gnb.Classes {
			logProb := math.Log(gnb.ClassPriors[class])
			for j := 0; j < len(x); j++ {
				mu := gnb.Means[class][j]
				variance := gnb.Variances[class][j] + gnb.Epsilon
				diff := x[j] - mu
				logProb += -0.3 * math.Log(2 * math.Pi * variance) - (diff * diff) / (2 * variance)
			}

			scores[class] = logProb
		}

		preds[i] = argmax(scores)
	}

	return preds
}

func (gnb *GaussianNB) Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64 {
	yPred := gnb.Predict(X)
	return metric(y, yPred)
}

func (gnb *GaussianNB) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(gnb)
}

func (gnb *GaussianNB) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewDecoder(f).Decode(gnb)
}

func (gnb *GaussianNB) GetParams() map[string]interface{} {
	return map[string]interface{}{}
}

func (gnb *GaussianNB) SetParams(map[string]interface{}) error {
	return nil
}

func argmax(m map[float64]float64) float64 {
	type kv struct {
		Key 	float64
		Value 	float64
	}

	var sorted []kv

	for k, v := range m {
		sorted = append(sorted, kv{k, v})
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Value > sorted[j].Value
	})

	return sorted[0].Key
}


var _ core.Model = (*GaussianNB)(nil)
var _ core.Scorable = (*GaussianNB)(nil)
var _ core.Serializable = (*GaussianNB)(nil)
var _ core.Params = (*GaussianNB)(nil)