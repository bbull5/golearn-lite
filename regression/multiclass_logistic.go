package regression

import (
	"encoding/gob"
	"os"

	"golearn-lite/matrix"
)


type MultiClassLogisticRegression struct {
	Classes				[]float64
	Classifiers			map[float64]*LogisticRegression
	LearningRate 		float64
	Iterations			int
}


func NewMultiClassLogisticRegression() *MultiClassLogisticRegression {
	return &MultiClassLogisticRegression{
		LearningRate: 0.1,
		Iterations: 1000,
		Classifiers: make(map[float64]*LogisticRegression),
	}
}

func (mc *MultiClassLogisticRegression) Fit(X matrix.Matrix, y []float64) error {
	classSet := make(map[float64]bool)
	for _, label := range y {
		classSet[label] = true
	}

	for class := range classSet {
		binaryY := make([]float64, len(y))
		for i, label := range y {
			if label == class {
				binaryY[i] = 1.0
			} else {
				binaryY[i] = 0.0
			}
		}

		clf := NewLogisticRegression()
		clf.LearningRate = mc.LearningRate
		clf.Iterations = mc.Iterations

		if err := clf.Fit(X, binaryY); err != nil {
			return err
		}

		mc.Classifiers[class] = clf
		mc.Classes = append(mc.Classes, class)
	}

	return nil
}

func (mc * MultiClassLogisticRegression) Predict(X matrix.Matrix) []float64 {
	n := X.Rows
	scores := make([][]float64, len(mc.Classes))

	for i, class := range mc.Classes {
		probs := mc.Classifiers[class].PredictClass(X)
		scores[i] = probs
	}

	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		maxProb := scores[0][i]
		bestClass := mc.Classes[0]
		for j := 1; j < len(mc.Classes); j++ {
			if scores[j][i] > maxProb {
				maxProb = scores[j][i]
				bestClass = mc.Classes[j]
			}
		}

		preds[i] = bestClass
	}

	return preds
}

func (mc *MultiClassLogisticRegression) Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64 {
	yPred := mc.Predict(X)
	return metric(y, yPred)
}

func (mc *MultiClassLogisticRegression) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(mc)
}

func (mc *MultiClassLogisticRegression) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewDecoder(f).Decode(mc)
}

func (mc *MultiClassLogisticRegression) GetParams() map[string]interface{} {
	return map[string]interface{} {
		"learning_rate": mc.LearningRate,
		"iterations": mc.Iterations,
	}
}

func (mc *MultiClassLogisticRegression) SetParams(params map[string]interface{}) error {
	if v, ok := params["learning_rate"].(float64); ok {
		mc.LearningRate = v
	}
	if v, ok := params["iterations"].(int); ok {
		mc.Iterations = v
	}

	return nil
}