package core

import "golearn-lite/matrix"

type Model interface {
	Fit(X matrix.Matrix, y []float64) error
	Predict(X matrix.Matrix) []float64
}

type Scorable interface {
	Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64
}


type Serializable interface {
	Save(path string) error
	Load(path string) error
}


type Params interface {
	GetParams() map[string]interface{}
	SetParams(params map[string]interface{}) error
}
