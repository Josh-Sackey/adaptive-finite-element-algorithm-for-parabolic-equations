# Allen-Cahn Example 4.7: implementation and paper comparison

## Purpose

This document records what the `allen-cahn-newton-nn` program solves, how the
finite element/Newton/neural-network pieces interact, what was obtained in the
deterministic reference run, and how that run compares with Example 4.7 of Hao,
Huang, Yi, and Yin (Journal of Computational Physics 565 (2026), 115173).

The implementation is a deal.II quadrilateral analogue of the paper's test. It
reproduces the model problem and principal physical behavior, but it is **not a
bit-for-bit reproduction of the paper's full triangular `hr`-adaptive
algorithm**. The distinctions are listed explicitly below.

## 1. Mathematical problem

The code solves the Allen-Cahn equation

\[
  u_t-\Delta u=\frac{u-u^3}{\varepsilon^2}
  \quad\text{in }\Omega=[-1,1]^2,
\]

with Dirichlet data

\[
  u=-1\quad\text{on }\partial\Omega
\]

and initial condition

\[
  u(x,y,0)=\tanh\left(
  \frac{0.25-\sqrt{(x-0.5)^2+(y-0.5)^2}}
       {\sqrt{2}\,\varepsilon}\right).
\]

Thus, the initial zero level set is a circle of radius `0.25` centered at
`(0.5,0.5)`. The parameters match Example 4.7:

| Parameter | Paper | This implementation |
|---|---:|---:|
| Domain | `[-1,1]^2` | `[-1,1]^2` |
| Interface parameter, epsilon | `0.01` | `0.01` |
| Time step, dt | `0.0003` | `0.0003` |
| Final time, T | `0.03` | `0.03` |
| Number of time steps | 100 | 100 |
| Boundary value | `-1` | `-1` |
| Nonlinear solver | Newton method | Newton method |
| Nominal estimator tolerance | `0.5` | `0.5` (applied to a relative estimator) |

## 2. Discretization and Newton formulation

The spatial approximation uses conforming bilinear Q1 finite elements on
quadrilateral cells. Backward Euler is used in time. Given the neural
approximation of the preceding solution, denoted here by
`u_previous`, the discrete nonlinear residual is

\[
  F(u;v)=\left(\frac{u-u_{\mathrm{previous}}}{\Delta t},v\right)
       +(\nabla u,\nabla v)
       -\left(\frac{u-u^3}{\varepsilon^2},v\right).
\]

For a Newton correction `delta_u`, the Jacobian is

\[
  F'(u)(\delta u,v)=
  \left(\frac{\delta u}{\Delta t},v\right)
  +(\nabla\delta u,\nabla v)
  +\left(\frac{3u^2-1}{\varepsilon^2}\delta u,v\right).
\]

At each Newton iteration the program solves

\[
  F'(u^k)(\delta u^k,v)=-F(u^k;v),
  \qquad u^{k+1}=u^k+\delta u^k.
\]

The correction has homogeneous boundary values, while the physical solution
has boundary value `-1`. Newton stops when the relative residual is at most
`1e-9` or the absolute residual is below `1e-11`.

## 3. Neural-network transfer

The neural network is used as a **mesh-free solution-transfer operator**. It
does not replace the finite element solve and it does not minimize the PDE
residual. After a finite element solution is available, the network is fitted
to nodal pairs `(x_i, u_h(x_i))`. At the following time step and after a mesh
change, its continuous output supplies the previous-time solution at points on
the current mesh.

The architecture is

```text
2 -> 40 -> 40 -> 40 -> 1
```

The two inputs are `(x,y)`, each hidden layer contains 40 neurons, and the
single output approximates the scalar phase field. Hidden layers use `tanh`;
the final layer is linear and has no bias. Training uses the limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer and mean squared error.
Weights are deterministically initialized with seed 42. The first fit is given
3000 optimizer iterations; later fits warm-start the same network and are given
100 iterations.

The boundary condition is built into the neural representation:

\[
  u_\theta(x,y)=(x^2-1)(y^2-1)N_\theta(x,y)-1.
\]

The multiplicative factor vanishes on every side of the square, so
`u_theta=-1` on the boundary for every possible set of network weights. This
matches the distance-function construction stated in the paper's Appendix A.

## 4. Adaptive mesh procedure

The initial mesh is an `8 x 8` quadrilateral grid. Five recovery-driven mesh
adaptations are performed at `t=0`. At later time levels the following loop is
used:

1. Evaluate the trained surrogate to initialize the Newton iterate and provide
   the preceding-time field.
2. Solve the nonlinear backward-Euler system by Newton's method.
3. Recover a continuous nodal gradient by averaging neighboring cell gradient
   values.
4. On every cell compute
   `eta_K = ||G(grad u_h)-grad u_h||_L2(K)`.
5. Compute `eta_rel = (sum_K eta_K^2)^(1/2) / |u_h|_H1`.
6. Stop adapting if `eta_rel <= 0.5`; otherwise mark the top 60 percent for
   refinement and bottom 40 percent for coarsening and repeat.

The maximum is seven adaptive solves per time level and the maximum mesh level
is eight.

## 5. Deterministic reference run

Reference output directory:

```text
runs/example-4p7-nn-deterministic
```

Run command:

```bash
./build/allen-cahn-newton-nn 0.03 runs/example-4p7-nn-deterministic 7
```

Summary:

| Quantity | Result |
|---|---:|
| Completed steps | 100 |
| VTU snapshots | 101 |
| Initial active cells / DoFs | 376 / 458 |
| Final active cells / DoFs | 559 / 685 |
| Maximum Newton corrections in one step | 8 |
| Maximum adaptive solves in one time level | 5 (at initialization) |
| Initial NN MSE | `7.15065e-6` |
| Largest NN MSE | `2.11193e-3` |
| Final NN MSE | `3.00723e-5` |
| Total NN training wall time | 123.055 s |
| Total run wall time | 175.366 s |
| Initial / final discrete energy | 185.588 / 141.020 |

Selected snapshots are shown below. Because `0.01` and `0.02` are not integer
multiples of `0.0003`, the closest stored time levels are `0.0099` and
`0.0201`.

| Time | Cells | DoFs | Newton corrections | Recovery estimator | Energy | Solution range |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 376 | 458 | 0 | 4.49854 | 185.588 | `[-1.0000, 1.0000]` |
| 0.0099 | 559 | 685 | 4 | 4.77928 | 143.932 | `[-1.0794, 1.0945]` |
| 0.0201 | 559 | 685 | 3 | 4.91189 | 140.479 | `[-1.0992, 1.0997]` |
| 0.0300 | 559 | 685 | 3 | 5.07231 | 141.020 | `[-1.0915, 1.0972]` |

The table reports the raw global recovery indicator stored in the CSV. The
stopping test uses its relative form, divided by the H1 seminorm, which was
about `0.44` at the final time.

## 6. Comparison with Figure 27 in the paper

Figure 27 presents the finite element solution and adaptive mesh at
`t=0.00`, `0.01`, `0.02`, and `0.03`. The paper's stated conclusion is
qualitative: refinement follows the moving sharp interface. It does not report
an exact solution, a table of DoFs, numerical errors, training time, or the
underlying Figure 27 data. Consequently, a defensible comparison is currently
qualitative rather than a claim of matching numerical error.

### Behaviors reproduced

- The initial interface is circular, centered at `(0.5,0.5)`, with radius
  `0.25`.
- The positive interior phase contracts under Allen-Cahn dynamics.
- Refinement remains concentrated around the thin transition layer rather than
  across the entire domain.
- The simulation reaches the same final time using the same `epsilon` and time
  step.
- Newton converges tightly at every time step.
- The neural transfer honors the boundary value exactly and warm-started
  training remains short compared with the initial fit.

### Important differences

| Paper algorithm | Current deal.II implementation |
|---|---|
| Triangular linear elements generated with Gmsh | Q1 quadrilateral elements in deal.II |
| Full `hr` adaptation with a mesh-size field | Hierarchical `h` refinement/coarsening |
| A fresh initial mesh is introduced within each time step | The adaptive mesh is carried through time |
| Up to seven prescribed adaptation stages, including least-squares prediction of target vertex count | Up to seven conventional solve-estimate-mark-adapt iterations |
| Neural surrogate transfers data between independently generated non-nested meshes | Neural surrogate transfers data to the current adapted quadrilateral mesh |
| Paper stopping statement uses global `eta <= 0.5` | Code uses relative `eta/|u_h|_H1 <= 0.5` |
| Paper describes a combined estimator that can retain information from the previous time level | Code estimates the current solution only |

These differences mean that matching the shape and location of the interface is
meaningful, whereas matching cell counts or mesh topology is not expected.

## 7. Interpretation and cautions

There is no manufactured exact solution for Example 4.7, so the present run
cannot report an exact L2 or H1 error. The strongest checks currently available
are Newton residual convergence, correct initial/boundary data, interface and
mesh behavior, comparison of selected snapshots with Figure 27, and refinement
studies in space and time.

The computed phase field overshoots the nominal physical interval `[-1,1]` by
roughly 0.10. The discrete energy falls substantially overall, but it is not
strictly monotone: 33 small increases occur during the 100 steps, with the
largest increase about `0.129`. These observations should be disclosed rather
than treated as proof of full agreement. Likely contributors include the coarse
Q1 mesh relative to the `O(epsilon)` interface thickness, neural-transfer
error, the relative stopping rule, and convergence to a Newton root that need
not be the energy-minimizing branch.

For a stronger publication-level comparison, the next tests should be:

1. Run a mesh/time-step refinement study and report interface radius, energy,
   overshoot, DoFs, and wall time.
2. Run a classical `SolutionTransfer` baseline under identical conditions to
   isolate neural-transfer error and cost.
3. Change the stopping test to the paper's raw estimator definition and examine
   the resulting mesh size.
4. Implement the paper's combined current/previous-time estimator.
5. If exact reproduction is required, add Gmsh triangular remeshing, mesh-size
   prediction, and the complete seven-stage `hr` algorithm.

## 8. Output files

- `allen-cahn.pvd`: ParaView time-series index.
- `allen-cahn-XXXX.vtu`: solution, NN previous solution, cell indicator, and
  refinement level at each stored time.
- `simulation-history.csv`: time, cells, DoFs, estimator, mass, energy, extrema,
  and cumulative wall time.
- `newton-history.csv`: absolute and relative residual at every Newton step.
- `neural-training.csv`: sample count, closure evaluations, final MSE, and
  training wall time.

Open `allen-cahn.pvd` in ParaView and use `solution` for the phase field. Use
`error_indicator` or `refinement_level` to inspect where adaptation occurs.

## Reference

J. Hao, Y. Huang, N. Yi, and P. Yin, *A neural network-enhanced hr-adaptive
finite element algorithm for parabolic equations*, Journal of Computational
Physics 565 (2026), 115173. Relevant material: recovery estimator in Section
2.3, neural transfer in Section 3.2.2, Algorithm 3, Example 4.7 and Figure 27,
and Appendix A.
