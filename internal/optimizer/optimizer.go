package optimizer

// Optimizer updates params in-place from grads. Clone resets state.
type Optimizer interface {
	Name() string
	Step(params, grads []float64)
	Clone() Optimizer
	Reset()
}
