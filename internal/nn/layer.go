package nn

import (
	"math"
	"math/rand"

	"github.com/shlaapnik/metopt-lab3/internal/optimizer"
)

// Layer is a fully-connected dense layer with activation.
type Layer struct {
	W      [][]float64 // [outDim][inDim]
	b      []float64   // [outDim]
	Act    Activation
	Lambda float64 // L2 regularization coefficient

	// cached during forward pass
	input  []float64
	Z      []float64 // pre-activation (exported for backward access from Network)
	output []float64

	// accumulated gradients, reset after each optimizer step
	dW [][]float64
	db []float64

	optW optimizer.Optimizer
	optB optimizer.Optimizer
}

func NewLayer(inDim, outDim int, act Activation, lambda float64, optProto optimizer.Optimizer) *Layer {
	l := &Layer{
		W:      make([][]float64, outDim),
		b:      make([]float64, outDim),
		Act:    act,
		Lambda: lambda,
		dW:     make([][]float64, outDim),
		db:     make([]float64, outDim),
		optW:   optProto.Clone(),
		optB:   optProto.Clone(),
	}
	// He initialization (suitable for ReLU; acceptable for Sigmoid too)
	scale := math.Sqrt(2.0 / float64(inDim))
	for i := range l.W {
		l.W[i] = make([]float64, inDim)
		l.dW[i] = make([]float64, inDim)
		for j := range l.W[i] {
			l.W[i][j] = rand.NormFloat64() * scale
		}
	}
	return l
}

func (l *Layer) Forward(x []float64) []float64 {
	l.input = x
	l.Z = make([]float64, len(l.W))
	for i, row := range l.W {
		s := l.b[i]
		for j, w := range row {
			s += w * x[j]
		}
		l.Z[i] = s
	}
	l.output = make([]float64, len(l.Z))
	for i, z := range l.Z {
		l.output[i] = l.Act.Forward(z)
	}
	return l.output
}

// BackwardDelta receives dL/dz for this layer, accumulates gradients,
// and returns dL/dx to propagate to the previous layer.
func (l *Layer) BackwardDelta(delta []float64) []float64 {
	for j, d := range delta {
		for k, x := range l.input {
			l.dW[j][k] += d * x
		}
		l.db[j] += d
	}
	dx := make([]float64, len(l.input))
	for j, d := range delta {
		for k := range l.input {
			dx[k] += l.W[j][k] * d
		}
	}
	return dx
}

// ApplyGradients averages accumulated gradients over batchSize,
// adds L2 penalty, then calls the optimizer.
func (l *Layer) ApplyGradients(batchSize int) {
	n := float64(batchSize)

	wFlat := flattenW(l.W)
	dwFlat := flattenW(l.dW)
	for i := range dwFlat {
		dwFlat[i] = dwFlat[i]/n + l.Lambda*wFlat[i]
	}
	l.optW.Step(wFlat, dwFlat)
	unflattenW(wFlat, l.W)

	for i := range l.db {
		l.db[i] /= n
	}
	l.optB.Step(l.b, l.db)

	for i := range l.dW {
		for j := range l.dW[i] {
			l.dW[i][j] = 0
		}
		l.db[i] = 0
	}
}

func flattenW(W [][]float64) []float64 {
	if len(W) == 0 {
		return nil
	}
	out := make([]float64, len(W)*len(W[0]))
	for i, row := range W {
		copy(out[i*len(row):], row)
	}
	return out
}

func unflattenW(flat []float64, W [][]float64) {
	if len(W) == 0 {
		return
	}
	cols := len(W[0])
	for i := range W {
		copy(W[i], flat[i*cols:(i+1)*cols])
	}
}
