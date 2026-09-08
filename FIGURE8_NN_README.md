# Neural-transfer rotating Gaussian on Q1 quadrilaterals

For a beginner-oriented explanation of the LibTorch implementation and how to
reuse it in another deal.II program, see
`NEURAL_NETWORK_IMPLEMENTATION_GUIDE.md`.

## Figure 8-aligned variant

This sibling project follows the coordinate convention visible in Figure 8
and used by the authors' released C++ example. The computational domain is
`[0,1]^2`, and the Gaussian center is
`(0.5 + 0.3 cos(2 pi t), 0.5 + 0.3 sin(2 pi t))`. Consequently, its center is
`(0.8,0.5)` at `t=0`.

The first four initial adaptations use recovery-guided fixed-fraction
quadrilateral refinement. At the fifth displayed adaptation, the code performs
additional recovery-guided bulk refinements until it reaches at least 12,818
Q1 degrees of freedom (subject to the level and iteration guards). This is the
hierarchical-square analogue of the paper's large SIZE/GENERATE jump; it is
not an unstructured triangular remesh.

The initial LibTorch L-BFGS fit is allowed up to 2,500 iterations, matching
the paper's observation that its first fit needs more than 2,000 epochs.
Warm-start fits at later time levels are allowed 100 iterations.

This sibling project implements a quadrilateral analogue of the neural-network
enhanced adaptive method in Hao, Huang, Yi, and Yin. It leaves the conventional
`fig8-quads` baseline unchanged.

Implemented components:

- Q1 quadrilateral FEM on `[0,1]^2` with backward Euler and `dt=0.01`.
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
