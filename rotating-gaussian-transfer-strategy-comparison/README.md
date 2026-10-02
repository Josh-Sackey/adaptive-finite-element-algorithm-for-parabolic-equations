# Rotating-Gaussian transfer-strategy comparison

This experiment compares three solution-transfer strategies using the same
Q1 rotating-Gaussian solver and adaptive settings:

1. `classical`: deal.II `SolutionTransfer`.
2. `nn-once-per-time`: train one neural surrogate for the solution at each
   time level and reuse the frozen network for every mesh crossing associated
   with the following backward-Euler step.
3. `nn-every-crossing`: experimental high-cost baseline that refits after
   every adaptive crossing.

The audited `rotating-gaussian-benchmark` already behaves like
`nn-once-per-time`: it trains at `t=0`, reuses that network through all mesh
crossings for the next time step, and trains once after obtaining each new
time-level solution. Therefore this directory both documents that fact and
provides a deliberately repeated-training baseline for measuring the cost.

The repeated-training mode is not the recommended parabolic algorithm. Once
it refits to the newly computed solution at the same time level, subsequent
adaptive solves no longer use exactly the same previous-time function in the
backward-Euler right-hand side. It is included to expose that cost and
algorithmic difference.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1
```

## Run all strategies

```bash
./run_comparison.sh 0.6 0.05 runs/comparison
```

Arguments are `top_fraction`, `final_time`, and `output_root`. After a short
validation run, use final time `1.0` for the full comparison.

Outputs include a wide `comparison.csv`, `training-summary.csv`, and either
`comparison.png` (with Matplotlib) or `comparison.svg` (dependency-free
fallback). Raw VTU/PVD run output remains ignored by the repository.

## Reproducible validation result

The committed short validation uses `top_fraction=0.6`, `dt=0.01`, and
`T=0.05`. Each method is launched in a separate process, and the NN seed is
set before layer construction so both NN strategies start identically.

| Method | NN fits | NN training (s) | Final L2 error | Final H1 seminorm | Final DoFs |
|---|---:|---:|---:|---:|---:|
| Classical | 0 | 0.00 | 1.292e-3 | 1.858e-1 | 13,684 |
| NN once per time level | 6 | 23.32 | 3.170e-3 | 9.760e-2 | 1,484 |
| NN every crossing | 36 | 29.63 | 6.410e-3 | 1.254e-1 | 1,344 |

These are validation data, not the final `T=1` study. They show that repeated
fitting created six times as many fits and did not improve either reported
error norm in this test. The classical and NN rows also use different mesh
evolution policies inherited from the Figure 8 benchmark, so runtime alone
must not be interpreted as a pure transfer-operator comparison.
