package metrics

// F1 computes the binary F1-score (positive class = 1).
func F1(preds, targets []int) float64 {
	tp, fp, fn := 0, 0, 0
	for i, p := range preds {
		t := targets[i]
		switch {
		case p == 1 && t == 1:
			tp++
		case p == 1 && t == 0:
			fp++
		case p == 0 && t == 1:
			fn++
		}
	}
	if tp == 0 {
		return 0
	}
	prec := float64(tp) / float64(tp+fp)
	rec := float64(tp) / float64(tp+fn)
	return 2 * prec * rec / (prec + rec)
}

// Accuracy returns fraction of correct predictions.
func Accuracy(preds, targets []int) float64 {
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
