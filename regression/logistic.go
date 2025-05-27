package regression

import (
	"encoding/gob"
	"errors"
	"math"
	"os"

	"golearn-lite/core"
	"golearn-lite/matrix"
)


type LogisticRegression struct {
	Coefficients []float64
	LearningRate float64
	Iterations int
}


func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{
		LearningRate: 0.1,
		Iterations: 1000,
	}
}

func(lr *LogisticRegression) Fit(X matrix.Matrix, y[]float64) error {
	if X.Rows != len(y) {
		return errors.New("number of samples in X and y do not match")
	}

	Xb := addBias(X)
	nSamples, nFeatures := Xb.Rows, Xb.Cols

	lr.Coefficients = make([]float64, nFeatures)

	for iter := 0; iter < lr.Iterations; iter++ {
		gradients := make([]float64, nFeatures)

		// Batch gradient descent
		for i := 0; i < nSamples; i++ {
			z := 0.0
			for j := 0; j < nFeatures; j++ {
				z += Xb.Data[i][j] * lr.Coefficients[j]
			}

			pred := sigmoid(z)
			err := pred - y[i]
			for j := 0; j < nFeatures; j++ {
				gradients[j] += err * Xb.Data[i][j]
			}
		}

		// Update coefficients
		for j := 0; j < nFeatures; j++ {
			lr.Coefficients[j] -= lr.LearningRate * gradients[j] / float64(nSamples)
		}
	}

	return nil
}

func (lr *LogisticRegression) Predict(X matrix.Matrix) []float64 {
	Xb := addBias(X)
	nSamples := Xb.Rows
	predictions := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		z := 0.0
		for j := 0; j < Xb.Cols; j++ {
			z += Xb.Data[i][j] * lr.Coefficients[j]
		}

		p := sigmoid(z)
		predictions[i] = p
	}

	return predictions
}

func (lr *LogisticRegression) Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64 {
	yPred := lr.Predict(X)
	return metric(y, yPred)
}

func (lr *LogisticRegression) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(lr)
}

func (lr *LogisticRegression) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewDecoder(f).Decode(lr)
}

func (lr *LogisticRegression) GetParams() map[string]interface{} {
	return map[string]interface{} {
		"learning_rate": lr.LearningRate,
		"iterations": lr.Iterations,
	}
}

func (lr *LogisticRegression) SetParams(params map[string]interface{}) error {
	if v, ok := params["learning_rate"].(float64); ok {
		lr.LearningRate = v
	}
	if v, ok := params["iterations"].(int); ok {
		lr.Iterations = v
	}

	return nil
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}


var _ core.Model = (*LogisticRegression)(nil)
var _ core.Scorable = (*LogisticRegression)(nil)
var _ core.Serializable = (*LogisticRegression)(nil)
var _ core.Params = (*LogisticRegression)(nil)