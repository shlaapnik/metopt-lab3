package main

import (
	"fmt"
	"os"

	"github.com/shlaapnik/metopt-lab3/internal/experiment"
)

func main() {
	if err := experiment.RunAll(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
