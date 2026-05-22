package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/shlaapnik/metopt-lab3/internal/experiment"
)

func main() {
	if err := experiment.RunAll(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	cmd := exec.Command("py", "plot_nn.py")
	cmd.Dir = "visualization"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "plots: %v\n", err)
	}
}
