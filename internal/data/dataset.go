package data

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

// Dataset holds feature matrix X and label vector Y.
type Dataset struct {
	X [][]float64
	Y []float64
}

// Load reads a CSV file where the last column is the target.
func Load(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	records, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("dataset %s: too few rows", path)
	}

	nCols := len(records[0])
	ds := &Dataset{}
	for _, row := range records[1:] {
		if len(row) != nCols {
			continue
		}
		feats := make([]float64, nCols-1)
		for j := 0; j < nCols-1; j++ {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, fmt.Errorf("col %d: %w", j, err)
			}
			feats[j] = v
		}
		t, err := strconv.ParseFloat(row[nCols-1], 64)
		if err != nil {
			return nil, fmt.Errorf("target: %w", err)
		}
		ds.X = append(ds.X, feats)
		ds.Y = append(ds.Y, t)
	}
	return ds, nil
}

// TrainTestSplit splits the dataset into train/test subsets.
func TrainTestSplit(ds *Dataset, trainRatio float64) (train, test *Dataset) {
	n := len(ds.X)
	idx := rand.Perm(n)
	trainN := int(float64(n) * trainRatio)

	train = &Dataset{}
	test = &Dataset{}
	for i, id := range idx {
		if i < trainN {
			train.X = append(train.X, ds.X[id])
			train.Y = append(train.Y, ds.Y[id])
		} else {
			test.X = append(test.X, ds.X[id])
			test.Y = append(test.Y, ds.Y[id])
		}
	}
	return
}

// Scaler performs standard (z-score) normalization fitted on training data.
type Scaler struct {
	Mean []float64
	Std  []float64
}

// FitScaler computes mean and std from X.
func FitScaler(X [][]float64) *Scaler {
	if len(X) == 0 {
		return &Scaler{}
	}
	dim := len(X[0])
	mean := make([]float64, dim)
	for _, x := range X {
		for j, v := range x {
			mean[j] += v
		}
	}
	n := float64(len(X))
	for j := range mean {
		mean[j] /= n
	}
	std := make([]float64, dim)
	for _, x := range X {
		for j, v := range x {
			d := v - mean[j]
			std[j] += d * d
		}
	}
	for j := range std {
		std[j] = math.Sqrt(std[j] / n)
		if std[j] < 1e-8 {
			std[j] = 1
		}
	}
	return &Scaler{Mean: mean, Std: std}
}

// Transform applies the fitted scaler to X.
func (s *Scaler) Transform(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, x := range X {
		row := make([]float64, len(x))
		for j, v := range x {
			row[j] = (v - s.Mean[j]) / s.Std[j]
		}
		out[i] = row
	}
	return out
}
