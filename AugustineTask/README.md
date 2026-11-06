# Allen-Cahn Newton solver with neural transfer

This project adapts the supplied stationary deal.II Newton example to Peimeng
et al., Example 4.7:

```text
u_t - Delta u = (u-u^3)/epsilon^2  on [-1,1]^2,
u = -1 on the boundary,
epsilon = 0.01, dt = 0.0003, T = 0.03.
```

The initial circular interface has radius 0.25 and center `(0.5,0.5)`. The
finite element method uses Q1 square elements and backward Euler. Newton's
method resolves the nonlinear equation at every time level. A
`[2,40,40,40,1]` LibTorch network is fitted to the preceding finite element
solution and supplies the previous-time value on each newly adapted mesh.
The physical NN output is

```text
(x^2-1)(y^2-1) N_theta(x,y) - 1,
```

so the Dirichlet condition is satisfied exactly.

## Build

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/deal.II-v9.7.1/lib/cmake/deal.II;$PWD/../rotating-gaussian-classical/third_party/libtorch"
cmake --build build -j2
```

## Run

The default command runs the complete paper interval with the NN surrogate:

```bash
./build/allen-cahn-newton-nn
```

Optional arguments are:

```text
allen-cahn-newton-nn [final_time] [output_directory] [max_adapt_iterations]
  [estimator_tolerance] [initial_subdivisions] [max_grid_level]
  [initial_adaptations] [time_step]
```

For a two-step smoke test:

```bash
./build/allen-cahn-newton-nn 0.0006 runs/smoke 2
```

Open `allen-cahn.pvd` in ParaView. The files contain the computed solution,
the NN representation of the previous-time solution, the gradient-recovery
indicator, and the refinement level. CSV files record Newton convergence,
NN training, and time-step diagnostics.

## Docker and Apptainer

For a reproducible local Docker image and an HPC-ready Apptainer image, follow
[`docs/HPC-CONTAINER-GUIDE.md`](docs/HPC-CONTAINER-GUIDE.md). The short path is:

```bash
./scripts/build-docker.sh
./scripts/run-docker.sh smoke
./scripts/export-docker.sh
./scripts/build-apptainer.sh
./scripts/run-apptainer.sh smoke
```

The recommended supercomputer experiment is the four-case convergence campaign
in `scripts/run-slurm.sh`, not merely a longer final time. It compares the
baseline with spatially refined, temporally refined, and combined production
runs under otherwise consistent conditions.

