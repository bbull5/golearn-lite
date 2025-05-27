package neighbors

import (
	"encoding/gob"
	"errors"
	"math"
	"os"
	"sort"

	"golearn-lite/core"
	"golearn-lite/matrix"
)


type KNN struct {
	XTrain 		[][]float64
	yTrain 		[]float64
	K			int
	Task		string
}

type neighbor struct {
	distance	float64
	label		float64
}


func NewKNN(k int, task string) *KNN {
	return &KNN {
		K:		k,
		Task:	task,
	}
}

func (knn *KNN) Fit(X matrix.Matrix, y []float64) error {
	if X.Rows != len(y) {
		return errors.New("number of samples in X and y must match")
	}

	knn.XTrain = X.Data
	knn.yTrain = y

	return nil
}

func (knn *KNN) Predict(X matrix.Matrix) []float64 {
	n := X.Rows
	preds := make([]float64, n)

	for i := 0; i < n; i++ {
		x := X.Data[i]
		neighbors := knn.getKNearest(x)

		if knn.Task == "classification" {
			preds[i] = majorityVote(neighbors)
		} else {
			preds[i] = meanVote(neighbors)
		}
	}

	return preds
}

func (knn *KNN) Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64 {
	yPred := knn.Predict(X)
	return metric(y, yPred)
}

func (knn *KNN) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(knn)
}

func (knn *KNN) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewDecoder(f).Decode(knn)
}

func (knn *KNN) GetParams() map[string]interface{} {
	return map[string]interface{} {
		"k": knn.K,
		"task": knn.Task,
	}
}

func (knn *KNN) SetParams(params map[string]interface{}) error {
	if v, ok := params["k"].(int); ok {
		knn.K = v
	}
	if v, ok := params["task"].(string); ok {
		knn.Task = v
	}

	return nil
}

func (knn *KNN) getKNearest(x []float64) []neighbor {
	all := make([]neighbor, len(knn.XTrain))

	for i, trainX := range knn.XTrain {
		dist := euclideanDisttance(x, trainX)
		all[i] = neighbor{distance: dist, label: knn.yTrain[i]}
	}

	sort.Slice(all, func(i, j int) bool {
		return all[i].distance < all[j].distance
	})

	return all[:knn.K]
}

func euclideanDisttance(a, b []float64) float64 {
	sum := 0.0

	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

func majorityVote(neighbors []neighbor) float64 {
	counts := make(map[float64]int)

	for _, n := range neighbors {
		counts[n.label]++
	}

	var bestLabel float64
	maxCount := -1

	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			bestLabel = label
		}
	}

	return bestLabel
}

func meanVote(neighbors []neighbor) float64 {
	sum := 0.0

	for _, n := range neighbors {
		sum += n.label
	}

	return sum / float64(len(neighbors))
}


var _ core.Model = (*KNN)(nil)
var _ core.Scorable = (*KNN)(nil)
var _ core.Serializable = (*KNN)(nil)
var _ core.Params = (*KNN)(nil)