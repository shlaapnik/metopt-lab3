package optimizer

import (
	"fmt"
	"math"
)

type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Eps   float64
	t     int
	m     []float64
	v     []float64
}

func NewAdam(lr, beta1, beta2, eps float64) *Adam {
	return &Adam{LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps}
}

func (a *Adam) Name() string {
	return fmt.Sprintf("Adam(lr=%.4f)", a.LR)
}

func (a *Adam) Step(params, grads []float64) {
	if a.m == nil {
		a.m = make([]float64, len(params))
		a.v = make([]float64, len(params))
	}
	a.t++
	bc1 := 1 - math.Pow(a.Beta1, float64(a.t))
	bc2 := 1 - math.Pow(a.Beta2, float64(a.t))
	for i, g := range grads {
		a.m[i] = a.Beta1*a.m[i] + (1-a.Beta1)*g
		a.v[i] = a.Beta2*a.v[i] + (1-a.Beta2)*g*g
		mHat := a.m[i] / bc1
		vHat := a.v[i] / bc2
		params[i] -= a.LR * mHat / (math.Sqrt(vHat) + a.Eps)
	}
}

func (a *Adam) Clone() Optimizer {
	return NewAdam(a.LR, a.Beta1, a.Beta2, a.Eps)
}

func (a *Adam) Reset() {
	a.t = 0
	a.m = nil
	a.v = nil
}
