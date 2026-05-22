package optimizer

// Optimizer updates a flat parameter slice in-place given its gradient.
// Clone returns a fresh instance with identical hyperparameters but zeroed state.
type Optimizer interface {
	Name() string
	Step(params, grads []float64)
	Clone() Optimizer
	Reset()
}
