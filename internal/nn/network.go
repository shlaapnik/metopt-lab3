package nn

import (
	"math"
	"math/rand"

	"github.com/shlaapnik/metopt-lab3/internal/optimizer"
)

// Config holds network and training hyperparameters.
type Config struct {
	HiddenSizes []int
	Lambda      float64 // L2 regularization coefficient
	BatchSize   int
	Epochs      int
	EarlyStopping int // patience in epochs; 0 = disabled
}

// Trace records per-epoch metrics for later analysis.
type Trace struct {
	Epoch     int
	TrainLoss float64
	TestLoss  float64
	TestAcc   float64
	TestF1    float64
}

// Network is a fully-connected neural network for binary classification.
type Network struct {
	layers []*Layer
	cfg    Config
}

// New builds a network with the given input dimension, config, and optimizer prototype.
// Each layer clones its own optimizer instances from optProto.
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

// Forward runs the network on one sample and returns the scalar output.
func (n *Network) Forward(x []float64) float64 {
	out := x
	for _, l := range n.layers {
		out = l.Forward(out)
	}
	return out[0]
}

// Backward computes gradients for one sample and accumulates them in each layer.
// Uses the numerically stable combined BCE+Sigmoid gradient: dL/dz_out = ŷ - y.
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

// TrainEpoch runs one epoch of mini-batch training and returns mean train loss.
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

// EvalLoss computes mean BCE loss on a dataset.
func (n *Network) EvalLoss(X [][]float64, y []float64) float64 {
	total := 0.0
	for i, x := range X {
		total += bceLoss(n.Forward(x), y[i])
	}
	return total / float64(len(X))
}

// PredictClass returns 0 or 1 for a single sample.
func (n *Network) PredictClass(x []float64) int {
	if n.Forward(x) >= 0.5 {
		return 1
	}
	return 0
}

// Train runs the full training loop with optional early stopping.
// It returns the full trace history.
func (n *Network) Train(
	trainX [][]float64, trainY []float64,
	testX [][]float64, testY []float64,
	evalFn func(preds, targets []int) float64,
) []Trace {
	var history []Trace
	bestTestLoss := math.MaxFloat64
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
		acc := accuracy(preds, targets)
		f1 := evalFn(preds, targets)

		history = append(history, Trace{
			Epoch:     epoch,
			TrainLoss: trainLoss,
			TestLoss:  testLoss,
			TestAcc:   acc,
			TestF1:    f1,
		})

		if n.cfg.EarlyStopping > 0 {
			if testLoss < bestTestLoss-1e-5 {
				bestTestLoss = testLoss
				noImprove = 0
			} else {
				noImprove++
				if noImprove >= n.cfg.EarlyStopping {
					break
				}
			}
		}
	}
	return history
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
