package optimizer

import "fmt"

type Momentum struct {
	LR  float64
	Mu  float64
	vel []float64
}

func NewMomentum(lr, mu float64) *Momentum {
	return &Momentum{LR: lr, Mu: mu}
}

func (m *Momentum) Name() string {
	return fmt.Sprintf("Momentum(lr=%.4f,mu=%.1f)", m.LR, m.Mu)
}

func (m *Momentum) Step(params, grads []float64) {
	if m.vel == nil {
		m.vel = make([]float64, len(params))
	}
	for i := range params {
		m.vel[i] = m.Mu*m.vel[i] - m.LR*grads[i]
		params[i] += m.vel[i]
	}
}

func (m *Momentum) Clone() Optimizer { return NewMomentum(m.LR, m.Mu) }
func (m *Momentum) Reset()           { m.vel = nil }
