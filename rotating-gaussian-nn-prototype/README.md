# Neural-transfer rotating Gaussian on Q1 quadrilaterals

This sibling project implements a quadrilateral analogue of the neural-network
enhanced adaptive method in Hao, Huang, Yi, and Yin. It leaves the conventional
`rotating-gaussian-classical` baseline unchanged.

Implemented components:

- Q1 quadrilateral FEM on `[-1,1]^2` with backward Euler and `dt=0.01`.
- Exact rotating-Gaussian forcing, initial data, and Dirichlet data.
- Weighted/recovered-gradient cell indicators.
- Five initial adaptations at `t=0`.
- LibTorch surrogate with published architecture `[2,40,40,40,1]`, `tanh`,
  Kaiming initialization, double precision, and L-BFGS optimization.
- Neural evaluation of the previous-time solution in the backward-Euler mass
  term, eliminating `SolutionTransfer` between time levels.
- Reset to the same coarse 8x8 quadrilateral mesh at each time level.
- Up to seven tolerance-controlled spatial adaptations per time level.
- VTU/PVD, FEM error history, and neural training history output.

Important method difference from the paper:

- The paper generates non-nested triangular meshes with Gmsh and a continuous
  size field. This project deliberately uses deal.II hierarchical quadrilateral
  refinement. It is therefore a quadrilateral analogue, not a bit-for-bit
  reproduction of the triangular mesh generator.

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1
```

Run a short validation (recommended first):

```bash
mkdir -p runs/validation
cd runs/validation
../../build/fig8-quads-nn 0.6 0.05
```

Run the full `T=1` problem:

```bash
mkdir -p runs/full
cd runs/full
../../build/fig8-quads-nn 0.6 1.0
```

The first argument is the top refinement fraction. The second is final time.
Outputs include `rotating-gaussian.pvd`, `simulation-history.csv`, and
`neural-training.csv`.
