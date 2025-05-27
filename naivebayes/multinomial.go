package naivebayes

import (
	"os"
	"errors"
	"math"
	"encoding/gob"

	"golearn-lite/core"
	"golearn-lite/matrix"
)


type MultinomialNB struct {
	ClassPriors			map[float64]float64
	FeatureLogProbs		map[float64][]float64	// P(x_j | c)
	Classes				[]float64
	Alpha				float64
}


func NewMultinomialNB() *MultinomialNB {
	return &MultinomialNB{
		ClassPriors: 		make(map[float64]float64),
		FeatureLogProbs: 	make(map[float64][]float64),
		Alpha:				1.0,
	}
}

func (mnb *MultinomialNB) Fit(X matrix.Matrix, y []float64) error {
	if X.Rows != len(y) {
		return errors.New("number of samples in X and y must match")
	}

	nSamples, nFeatures := X.Rows, X.Cols
	classCounts := make(map[float64]int)
	classSums := make(map[float64][]float64)
	classSet := make(map[float64]bool)

	for i, row := range X.Data {
		label := y[i]
		classSet[label] = true
		classCounts[label]++

		if _, exists := classSums[label]; !exists {
			classSums[label] = make([]float64, nFeatures)
		}

		for j := 0; j < nFeatures; j++ {
			classSums[label][j] += row[j]
		}
	}

	for class := range classSet {
		mnb.Classes = append(mnb.Classes, class)
	}

	for _, class := range mnb.Classes {
		sums := classSums[class]
		total := 0.0
		for _, val := range sums {
			total += val
		}
		total += mnb.Alpha * float64(nFeatures)

		logProbs := make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			num := sums[j] + mnb.Alpha
			logProbs[j] = math.Log(num / total)
		}

		mnb.FeatureLogProbs[class] = logProbs
		mnb.ClassPriors[class] = float64(classCounts[class]) / float64(nSamples)
	}

	return nil
}

func (mnb *MultinomialNB) Predict(X matrix.Matrix) []float64 {
	n := X.Rows
	preds := make([]float64, n)

	for i := 0; i < n; i++ {
		x := X.Data[i]
		scores := make(map[float64]float64)

		for _, class := range mnb.Classes {
			logProb := math.Log(mnb.ClassPriors[class])
			for j := 0; j < len(x); j++ {
				logProb += x[j] * mnb.FeatureLogProbs[class][j]
			}
			scores[class] = logProb
		}

		preds[i] = argmax(scores)
	}

	return preds
}

func (mnb *MultinomialNB) Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64 {
	yPred := mnb.Predict(X)
	return metric(y, yPred)
}

func (mnb *MultinomialNB) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(mnb)
}

func (mnb *MultinomialNB) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewDecoder(f).Decode(mnb)
}

func (mnb *MultinomialNB) GetParams() map[string]interface{} {
	return map[string]interface{} {
		"alpha": mnb.Alpha,
	}
}

func (mnb *MultinomialNB) SetParams(params map[string]interface{}) error {
	if val, ok := params["alpha"].(float64); ok {
		mnb.Alpha = val
	}

	return nil
}


var _ core.Model = (*MultinomialNB)(nil)
var _ core.Scorable = (*MultinomialNB)(nil)
var _ core.Serializable = (*MultinomialNB)(nil)
var _ core.Params = (*MultinomialNB)(nil)


