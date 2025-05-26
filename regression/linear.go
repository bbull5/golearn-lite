package regression

import (
	"encoding/gob"
	"errors"
	"os"

	// "golearn-lite/core"
	"golearn-lite/matrix"
)

type LinearRegression struct {
	Coefficients []float64
}

func NewLinearRegression() *LinearRegression {
	return &LinearRegression{}
}

func (lr *LinearRegression) Fit(X matrix.Matrix, y []float64) error {
	if X.Rows != len(y) {
		return errors.New("number of samples in X and y do not match")
	}

	// Add bias term
	Xb := addBias(X)

	// Convert y to matrix
	yMat := make([][]float64, len(y))

	for i := range y {
		yMat[i] = []float64{y[i]}
	}
	Y := matrix.New(yMat)

	XT := Xb.Transpose()
	XTX, _ := XT.Dot(Xb)
	XTX_inv, err := XTX.Inverse()
	if err != nil {
		return err
	}

	XTy, _ := XT.Dot(Y)
	theta, _ := XTX_inv.Dot(XTy)

	// Extract coefficients
	lr.Coefficients = make([]float64, theta.Rows)
	for i := range theta.Data {
		lr.Coefficients[i] = theta.Data[i][0]
	}

	return nil
}

func (lr *LinearRegression) Predict(X matrix.Matrix) []float64 {
	Xb := addBias(X)
	pred := make([]float64, Xb.Rows)

	for i := 0; i < Xb.Rows; i++ {
		sum := 0.0
		for j := 0; j < Xb.Cols; j++ {
			sum += Xb.Data[i][j] * lr.Coefficients[j]
		}
		pred[i] = sum
	}

	return pred
}

func (lr *LinearRegression) Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64 {
	yPred := lr.Predict(X)
	return metric(y, yPred)
}

func (lr *LinearRegression) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(lr.Coefficients)
}

func (lr *LinearRegression) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewDecoder(f).Decode(&lr.Coefficients)
}

func (lr* LinearRegression) GetParams() map[string]interface{} {
	return map[string]interface{}{}
}

func (lr *LinearRegression) SetParams(params map[string]interface{}) error {
	return nil
}

func addBias(X matrix.Matrix) matrix.Matrix {
	biased := make([][]float64, X.Rows)

	for i := range X.Data {
		row := make([]float64, X.Cols+1)
		row[0] = 1.0
		copy(row[1:], X.Data[i])
		biased[i] = row
	}

	return matrix.New(biased)
}
