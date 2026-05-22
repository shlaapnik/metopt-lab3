package experiment

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/shlaapnik/metopt-lab3/internal/data"
	"github.com/shlaapnik/metopt-lab3/internal/metrics"
	"github.com/shlaapnik/metopt-lab3/internal/nn"
	"github.com/shlaapnik/metopt-lab3/internal/optimizer"
)

const (
	outDir = "out"
	seed   = 666
)

var weights = map[string]float64{"dataset1": 0.3, "dataset2": 0.3, "dataset3": 0.4}

func RunAll() error {
	os.RemoveAll(outDir)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	paths, err := filepath.Glob("dataset*.csv")
	if err != nil {
		return err
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return fmt.Errorf("no dataset*.csv in current dir")
	}

	opts := []optimizer.Optimizer{
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

	f1 := map[string]float64{}
	for _, path := range paths {
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))

		ds, err := data.Load(path)
		if err != nil {
			return err
		}

		rand.Seed(seed)
		train, test := data.TrainTestSplit(ds, 0.8)
		sc := data.FitScaler(train.X)
		trainX, testX := sc.Transform(train.X), sc.Transform(test.X)

		fmt.Printf("%s  %dx%d\n", name, len(ds.X), len(ds.X[0]))
		for _, opt := range opts {
			rand.Seed(seed)
			net := nn.New(len(ds.X[0]), cfg, opt)
			hist := net.Train(trainX, train.Y, testX, test.Y, metrics.F1)

			b := bestEpoch(hist)
			if b.TestF1 > f1[name] {
				f1[name] = b.TestF1
			}
			fmt.Printf("  %-26s ep %3d  loss %.4f  acc %.3f  f1 %.3f\n",
				opt.Name(), b.Epoch, b.TrainLoss, b.TestAcc, b.TestF1)

			out := filepath.Join(outDir, name+"__"+safeName(opt.Name())+".csv")
			if err := saveHistory(out, hist); err != nil {
				return err
			}
		}
		fmt.Println()
	}

	printScore(f1)
	return nil
}

func bestEpoch(hist []nn.Trace) nn.Trace {
	best := hist[0]
	for _, t := range hist[1:] {
		if t.TestLoss < best.TestLoss {
			best = t
		}
	}
	return best
}

func printScore(f1 map[string]float64) {
	total := 0.0
	for _, name := range []string{"dataset1", "dataset2", "dataset3"} {
		v, ok := f1[name]
		if !ok {
			fmt.Printf("%s f1 ---- (нет файла)\n", name)
			continue
		}
		fmt.Printf("%s f1 %.3f\n", name, v)
		total += weights[name] * v
	}
	fmt.Printf("score = 0.3*d1 + 0.3*d2 + 0.4*d3 = %.3f\n", total)
}

func saveHistory(path string, hist []nn.Trace) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	w.Write([]string{"epoch", "train_loss", "test_loss", "test_acc", "test_f1"})
	for _, t := range hist {
		w.Write([]string{
			strconv.Itoa(t.Epoch),
			strconv.FormatFloat(t.TrainLoss, 'f', 6, 64),
			strconv.FormatFloat(t.TestLoss, 'f', 6, 64),
			strconv.FormatFloat(t.TestAcc, 'f', 6, 64),
			strconv.FormatFloat(t.TestF1, 'f', 6, 64),
		})
	}
	return nil
}

func safeName(s string) string {
	r := strings.NewReplacer("(", "_", ")", "", "=", "", ",", "_", " ", "")
	return r.Replace(s)
}
