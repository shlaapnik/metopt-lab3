package optimizer

import "fmt"

type SGD struct {
	LR float64
}

func NewSGD(lr float64) *SGD { return &SGD{LR: lr} }

func (s *SGD) Name() string { return fmt.Sprintf("SGD(lr=%.4f)", s.LR) }

func (s *SGD) Step(params, grads []float64) {
	for i := range params {
		params[i] -= s.LR * grads[i]
	}
}

func (s *SGD) Clone() Optimizer { return NewSGD(s.LR) }
func (s *SGD) Reset()           {}
