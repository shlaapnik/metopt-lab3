package nn

import "math"

type Activation interface {
	Forward(z float64) float64
	Backward(z float64) float64
}

type ReLU struct{}

func (ReLU) Forward(z float64) float64 {
	if z > 0 {
		return z
	}
	return 0
}

func (ReLU) Backward(z float64) float64 {
	if z > 0 {
		return 1
	}
	return 0
}

type Sigmoid struct{}

func (Sigmoid) Forward(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (Sigmoid) Backward(z float64) float64 {
	s := Sigmoid{}.Forward(z)
	return s * (1 - s)
}
