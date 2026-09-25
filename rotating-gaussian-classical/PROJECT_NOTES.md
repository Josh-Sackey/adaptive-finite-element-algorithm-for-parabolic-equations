# Project notes and deferred work

## Next phase: neural-network-enhanced simulation

The next major task is to add a LibTorch neural-network component to the
adaptive rotating-Gaussian simulation, following the approach used in the
authors' repository:

<https://github.com/FEMmaster/NN-enhanced-hr-AFEA-for-parabolic-equations>

This work is intentionally deferred. The current conventional deal.II solver
and its top-fraction experiments must remain available as validated baselines.

### Intended scope

- Study the repository's 2D neural-network implementation and map its inputs,
  outputs, normalization, architecture, loss, training schedule, and inference
  stage to the current Q1 quadrilateral solver.
- Use the existing project-local LibTorch 2.7.1 installation and the verified
  `torch-smoke-test` target.
- Feed appropriate finite-element and gradient-recovery data to the network.
- Compare network-guided adaptation with the conventional
  `GridRefinement::refine_and_coarsen_fixed_fraction` baselines.
- Preserve the same PDE, exact solution, time step, initial mesh, level limits,
  error norms, and output cadence so comparisons remain meaningful.
- Record active cells, degrees of freedom, L2 error, H1 seminorm error,
  training/inference cost, and any network-specific metrics.
- Keep neural-network code in separate source/header files instead of placing
  the entire implementation in `fig8-quads.cc`.

### Baselines to retain

The conventional simulations use top fractions 0.1 through 0.6 and bottom
fractions equal to `1 - top_fraction`. Their histories are stored under:

```text
runs/comparison-csv
```

The Step-26 baseline is:

```text
top_fraction = 0.6
bottom_fraction = 0.4
```

### Out of scope until the NN phase begins

- Do not change the validated conventional results merely to accommodate the
  network.
- Do not implement r-adaptation until its role and required data have been
  mapped from the paper and repository.
- Do not assume that the triangular-mesh implementation can be copied
  directly; it must be adapted carefully to Q1 quadrilateral cells.
