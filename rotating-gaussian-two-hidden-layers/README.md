# Rotating-Gaussian timestep-refinement study

This experiment tests whether backward-Euler temporal error explains the
larger early-time NN `L2` error observed before `t=0.05`. It leaves the
validated transfer-strategy comparison unchanged.

The controlled sweep compares classical deal.II `SolutionTransfer` with the
NN surrogate trained once per time level at:

- `dt=0.01` (baseline; 5 steps to `T=0.05`),
- `dt=0.005` (10 steps), and
- `dt=0.0025` (20 steps).

All spatial, marking, estimator, network, and optimization settings are held
fixed. The classical mesh-adaptation interval is fixed at `0.05` units of
physical time instead of every five steps, preventing timestep refinement from
silently changing its adaptation frequency. Because a smaller timestep creates
more NN fits, the study reports both accuracy and cost.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1
./run_timestep_study.sh 0.6 0.05 runs/early-time
```

The executable arguments are:

```text
MODE TOP_FRACTION FINAL_TIME TIME_STEP OUTPUT_ROOT
```

Curated outputs are `timestep-history.csv`, `timestep-summary.csv`, and
`timestep-errors.png`. Raw VTU/PVD and per-run histories remain under each
timestep directory for inspection and should remain Git-ignored.
