package trees

import (
	"os"
	"errors"
	"math"
	"encoding/gob"

	"golearn-lite/core"
	"golearn-lite/matrix"
)


func init() {
	gob.Register(&TreeNode{})
}


type DecisionTree struct {
	Root     *TreeNode
	MaxDepth int
	MinSize  int
	Task     string // "classification" or "regression"
}

type TreeNode struct {
	FeatureIndex int
	Threshold    float64
	Left         *TreeNode
	Right        *TreeNode
	Value        float64
	ClassCounts  map[float64]int
	IsLeaf       bool
}

func NewDecisionTree(task string) *DecisionTree {
	return &DecisionTree{
		MaxDepth: 10,
		MinSize:  2,
		Task:     task,
	}
}

func (dt *DecisionTree) Fit(X matrix.Matrix, y []float64) error {
	if X.Rows != len(y) {
		return errors.New("number of rows in X must equal length of y")
	}

	dt.Root = dt.buildTree(X.Data, y, 0)
	return nil
}

func (dt *DecisionTree) buildTree(X [][]float64, y []float64, depth int) *TreeNode {
	if len(y) == 0 {
		return nil
	}

	// leaf condition
	if depth >= dt.MaxDepth || len(y) <= dt.MinSize || dt.isPure(y) {
		if dt.Task == "classification" {
			return dt.makeClassificationLeaf(y)
		}

		return dt.makeRegressionLeaf(y)
	}

	// Best split
	bestIdx, bestThresh, _, leftX, rightX, leftY, rightY := dt.bestSplit(X, y)
	if bestIdx == -1 {
		if dt.Task == "classification" {
			return dt.makeClassificationLeaf(y)
		}

		return dt.makeRegressionLeaf(y)
	}

	return &TreeNode{
		FeatureIndex: bestIdx,
		Threshold:    bestThresh,
		Left:         dt.buildTree(leftX, leftY, depth+1),
		Right:        dt.buildTree(rightX, rightY, depth+1),
	}
}

func (dt *DecisionTree) Predict(X matrix.Matrix) []float64 {
	preds := make([]float64, X.Rows)

	for i, row := range X.Data {
		preds[i] = dt.predictOne(dt.Root, row)
	}

	return preds
}

func (dt *DecisionTree) Score(X matrix.Matrix, y []float64, metric func(yTrue, yPred []float64) float64) float64 {
	yPred := dt.Predict(X)
	return metric(y, yPred)
}

func (dt *DecisionTree) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewEncoder(f).Encode(dt)
}

func (dt *DecisionTree) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return gob.NewDecoder(f).Decode(dt)
}

func (dt *DecisionTree) GetParams() map[string]interface{} {
	return map[string]interface{} {
		"max_depth":	dt.MaxDepth,
		"min_size":		dt.MinSize,
		"task":			dt.Task,
	}
}

func (dt *DecisionTree) SetParams(params map[string]interface{}) error {
	if v, ok := params["max_depth"].(int); ok {
		dt.MaxDepth = v
	}
	if v, ok := params["min_size"].(int); ok {
		dt.MinSize = v
	}
	if v, ok := params["task"].(string); ok {
		dt.Task = v
	}

	return nil
}

func (dt *DecisionTree) bestSplit(X [][]float64, y []float64) (
	bestIdx int,
	bestThresh float64,
	bestScore float64,
	leftX, rightX [][]float64,
	leftY, rightY []float64,
) {
	nFeatures := len(X[0])
	bestScore = math.Inf(1)
	bestIdx = -1

	for featureIdx := 0; featureIdx < nFeatures; featureIdx++ {
		thresholds := uniqueValues(X, featureIdx)

		for _, threshold := range thresholds {
			currLeftX, currRightX, currLeftY, currRightY := splitDataset(X, y, featureIdx, threshold)
			if len(currLeftY) == 0 || len(currRightY) == 0 {
				continue
			}

			var score float64

			if dt.Task == "classification" {
				score = gini(currLeftY, currRightY)
			} else {
				score = mse(currLeftY, currRightY)
			}

			if score < bestScore {
				bestScore = score
				bestIdx = featureIdx
				bestThresh = threshold
				leftX, rightX = currLeftX, currRightX
				leftY, rightY = currLeftY, currRightY
			}
		}
	}

	return
}

func (dt *DecisionTree) predictOne(node *TreeNode, x []float64) float64 {
	if node.IsLeaf {
		if dt.Task == "classification" {
			return majorityLabel(node.ClassCounts)
		}

		return node.Value
	}

	if x[node.FeatureIndex] < node.Threshold {
		return dt.predictOne(node.Left, x)
	}

	return dt.predictOne(node.Right, x)
}

func (dt *DecisionTree) makeRegressionLeaf(y []float64) *TreeNode {
	sum := 0.0

	for _, val := range y {
		sum += val
	}

	return &TreeNode{
		IsLeaf: true,
		Value:  sum / float64(len(y)),
	}
}

func (dt *DecisionTree) makeClassificationLeaf(y []float64) *TreeNode {
	counts := make(map[float64]int)

	for _, label := range y {
		counts[label]++
	}

	return &TreeNode{
		IsLeaf:      true,
		ClassCounts: counts,
	}
}

func majorityLabel(counts map[float64]int) float64 {
	var bestLabel float64
	max := -1

	for label, count := range counts {
		if count > max {
			max = count
			bestLabel = label
		}
	}

	return bestLabel
}

func (dt *DecisionTree) isPure(y []float64) bool {
	first := y[0]

	for _, val := range y {
		if val != first {
			return false
		}
	}

	return true
}

func splitDataset(X [][]float64, y []float64, featureIdx int, threshold float64) (
	leftX, rightX [][]float64,
	leftY, rightY []float64,
) {
	for i, row := range X {
		if row[featureIdx] < threshold {
			leftX = append(leftX, row)
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, row)
			rightY = append(rightY, y[i])
		}
	}

	return
}

func uniqueValues(X [][]float64, featureIdx int) []float64 {
	seen := make(map[float64]bool)

	for _, row := range X {
		seen[row[featureIdx]] = true
	}

	out := make([]float64, 0, len(seen))

	for val := range seen {
		out = append(out, val)
	}

	return out
}

func gini(yLeft, yRight []float64) float64 {
	total := float64(len(yLeft) + len(yRight))

	return (float64(len(yLeft))/total)*giniImpurity(yLeft) + (float64(len(yRight))/total)*giniImpurity(yRight)
}

func giniImpurity(y []float64) float64 {
	counts := make(map[float64]int)

	for _, label := range y {
		counts[label]++
	}

	impurity := 1.0
	n := float64(len(y))

	for _, count := range counts {
		p := float64(count) / n
		impurity -= p * p
	}

	return impurity
}

func mse(yLeft, yRight []float64) float64 {
	weighted := func(y []float64) float64 {
		mean := 0.0

		for _, v := range y {
			mean += v
		}

		mean /= float64(len(y))
		err := 0.0

		for _, v := range y {
			diff := v - mean
			err += diff * diff
		}

		return err
	}

	return weighted(yLeft) + weighted(yRight)
}


var _ core.Model = (*DecisionTree)(nil)
var _ core.Scorable = (*DecisionTree)(nil)
var _ core.Serializable = (*DecisionTree)(nil)
var _ core.Params = (*DecisionTree)(nil)
