package experiment

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"

	"github.com/shlaapnik/metopt-lab3/internal/data"
	"github.com/shlaapnik/metopt-lab3/internal/metrics"
	"github.com/shlaapnik/metopt-lab3/internal/nn"
	"github.com/shlaapnik/metopt-lab3/internal/optimizer"
)

const outDir = "out"

// RunAll loads both datasets, trains each optimizer on each, prints results, saves CSVs.
func RunAll() error {
	fmt.Println("=== metopt-lab3: нейронная сеть для классификации ===")
	fmt.Println()

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	datasets := []struct {
		name string
		path string
	}{
		{"dataset1", "dataset1.csv"},
		{"dataset2", "dataset2.csv"},
	}

	optimizers := []optimizer.Optimizer{
		optimizer.NewSGD(0.01),
		optimizer.NewMomentum(0.005, 0.9),
		optimizer.NewAdam(0.001, 0.9, 0.999, 1e-8),
	}

	cfg := nn.Config{
		HiddenSizes:   []int{32, 16},
		Lambda:        0.001,
		BatchSize:     32,
		Epochs:        300,
		EarlyStopping: 30,
	}

	for _, ds := range datasets {
		fmt.Printf("--- %s ---\n", ds.name)

		raw, err := data.Load(ds.path)
		if err != nil {
			return fmt.Errorf("load %s: %w", ds.path, err)
		}

		rand.Seed(42)
		train, test := data.TrainTestSplit(raw, 0.8)

		scaler := data.FitScaler(train.X)
		trainX := scaler.Transform(train.X)
		testX := scaler.Transform(test.X)

		inputDim := len(raw.X[0])

		for _, opt := range optimizers {
			rand.Seed(42)
			net := nn.New(inputDim, cfg, opt)

			history := net.Train(trainX, train.Y, testX, test.Y, metrics.F1)

			last := history[len(history)-1]
			fmt.Printf("  %-30s  epochs=%3d  train_loss=%.4f  test_acc=%.4f  F1=%.4f\n",
				opt.Name(), last.Epoch, last.TrainLoss, last.TestAcc, last.TestF1)

			csvPath := filepath.Join(outDir, fmt.Sprintf("%s__%s.csv", ds.name, safeName(opt.Name())))
			if err := saveHistory(csvPath, history); err != nil {
				fmt.Printf("    warn: cannot save %s: %v\n", csvPath, err)
			}
		}
		fmt.Println()
	}
	return nil
}

func saveHistory(path string, history []nn.Trace) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	_ = w.Write([]string{"epoch", "train_loss", "test_loss", "test_acc", "test_f1"})
	for _, t := range history {
		_ = w.Write([]string{
			strconv.Itoa(t.Epoch),
			strconv.FormatFloat(t.TrainLoss, 'f', 6, 64),
			strconv.FormatFloat(t.TestLoss, 'f', 6, 64),
			strconv.FormatFloat(t.TestAcc, 'f', 6, 64),
			strconv.FormatFloat(t.TestF1, 'f', 6, 64),
		})
	}
	return nil
}

func safeName(name string) string {
	out := make([]rune, 0, len(name))
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			out = append(out, r)
		case r == '-' || r == '_':
			out = append(out, r)
		default:
			out = append(out, '_')
		}
	}
	return string(out)
}
