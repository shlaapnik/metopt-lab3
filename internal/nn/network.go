package nn

import (
	"math"
	"math/rand"

	"github.com/shlaapnik/metopt-lab3/internal/optimizer"
)

type Config struct {
	HiddenSizes   []int
	Lambda        float64
	BatchSize     int
	Epochs        int
	EarlyStopping int
}

type Trace struct {
	Epoch     int
	TrainLoss float64
	TestLoss  float64
	TestAcc   float64
	TestF1    float64
}

type Network struct {
	layers []*Layer
	cfg    Config
}

// New builds inputDim -> hidden (ReLU) -> 1 (Sigmoid).
func New(inputDim int, cfg Config, optProto optimizer.Optimizer) *Network {
	sizes := make([]int, 0, len(cfg.HiddenSizes)+2)
	sizes = append(sizes, inputDim)
	sizes = append(sizes, cfg.HiddenSizes...)
	sizes = append(sizes, 1)

	n := &Network{cfg: cfg}
	for i := 0; i < len(sizes)-1; i++ {
		var act Activation
		if i == len(sizes)-2 {
			act = Sigmoid{}
		} else {
			act = ReLU{}
		}
		n.layers = append(n.layers, NewLayer(sizes[i], sizes[i+1], act, cfg.Lambda, optProto))
	}
	return n
}

func (n *Network) Forward(x []float64) float64 {
	out := x
	for _, l := range n.layers {
		out = l.Forward(out)
	}
	return out[0]
}

// Backward: output delta is the combined BCE+sigmoid gradient ŷ - y.
func (n *Network) Backward(target float64) {
	last := n.layers[len(n.layers)-1]
	delta := []float64{last.output[0] - target}

	for i := len(n.layers) - 1; i >= 0; i-- {
		dx := n.layers[i].BackwardDelta(delta)
		if i > 0 {
			prev := n.layers[i-1]
			delta = make([]float64, len(dx))
			for k, d := range dx {
				delta[k] = d * prev.Act.Backward(prev.Z[k])
			}
		}
	}
}

func (n *Network) TrainEpoch(X [][]float64, y []float64) float64 {
	indices := rand.Perm(len(X))
	totalLoss := 0.0
	bsz := n.cfg.BatchSize

	for start := 0; start < len(X); start += bsz {
		end := start + bsz
		if end > len(X) {
			end = len(X)
		}
		batch := indices[start:end]
		for _, idx := range batch {
			pred := n.Forward(X[idx])
			totalLoss += bceLoss(pred, y[idx])
			n.Backward(y[idx])
		}
		for _, l := range n.layers {
			l.ApplyGradients(len(batch))
		}
	}
	return totalLoss / float64(len(X))
}

func (n *Network) EvalLoss(X [][]float64, y []float64) float64 {
	total := 0.0
	for i, x := range X {
		total += bceLoss(n.Forward(x), y[i])
	}
	return total / float64(len(X))
}

func (n *Network) PredictClass(x []float64) int {
	if n.Forward(x) >= 0.5 {
		return 1
	}
	return 0
}

// Train loops with early stopping, then restores the best (lowest test-loss) weights.
func (n *Network) Train(
	trainX [][]float64, trainY []float64,
	testX [][]float64, testY []float64,
	evalFn func(preds, targets []int) float64,
) []Trace {
	var history []Trace
	bestTestLoss := math.MaxFloat64
	var bestW [][][]float64
	var bestB [][]float64
	noImprove := 0

	for epoch := 1; epoch <= n.cfg.Epochs; epoch++ {
		trainLoss := n.TrainEpoch(trainX, trainY)
		testLoss := n.EvalLoss(testX, testY)

		preds := make([]int, len(testX))
		targets := make([]int, len(testY))
		for i, x := range testX {
			preds[i] = n.PredictClass(x)
			targets[i] = int(testY[i])
		}

		history = append(history, Trace{
			Epoch:     epoch,
			TrainLoss: trainLoss,
			TestLoss:  testLoss,
			TestAcc:   accuracy(preds, targets),
			TestF1:    evalFn(preds, targets),
		})

		if testLoss < bestTestLoss {
			bestTestLoss = testLoss
			bestW, bestB = n.snapshotParams()
			noImprove = 0
		} else {
			noImprove++
			if n.cfg.EarlyStopping > 0 && noImprove >= n.cfg.EarlyStopping {
				break
			}
		}
	}

	if bestW != nil {
		n.restoreParams(bestW, bestB)
	}
	return history
}

func (n *Network) snapshotParams() ([][][]float64, [][]float64) {
	w := make([][][]float64, len(n.layers))
	b := make([][]float64, len(n.layers))
	for i, l := range n.layers {
		w[i], b[i] = l.SnapshotParams()
	}
	return w, b
}

func (n *Network) restoreParams(w [][][]float64, b [][]float64) {
	for i, l := range n.layers {
		l.RestoreParams(w[i], b[i])
	}
}

func bceLoss(pred, target float64) float64 {
	const eps = 1e-12
	if pred < eps {
		pred = eps
	}
	if pred > 1-eps {
		pred = 1 - eps
	}
	return -(target*math.Log(pred) + (1-target)*math.Log(1-pred))
}

func accuracy(preds, targets []int) float64 {
	if len(preds) == 0 {
		return 0
	}
	correct := 0
	for i, p := range preds {
		if p == targets[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(preds))
}
