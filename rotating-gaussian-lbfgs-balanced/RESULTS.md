# L-BFGS convergence-setting study

## Fixed problem and network

All runs use the selected `2-35-35-35-1` network, seed 42, strong-Wolfe line
search, Q1 quadrilateral FEM, the same adaptive policy, and one neural fit per
accepted time level. Only the L-BFGS convergence settings change.

| Setting | Initial/later budgets | Gradient tolerance | Change tolerance | History |
|---|---:|---:|---:|---:|
| Strict baseline | 2500 / 100 | 1e-9 | 1e-12 | 100 |
| Balanced | 2100 / 90 | 1e-8 | 1e-11 | 75 |
| Moderate | 1800 / 75 | 1e-7 | 1e-10 | 50 |
| Aggressive | 1200 / 50 | 1e-6 | 1e-8 | 25 |

## Screening at dt = 0.0025

| Setting | Evaluations | Training (s) | L2 error | H1 seminorm | DoFs |
|---|---:|---:|---:|---:|---:|
| Strict | 4,626 | 31.940 | 5.929339e-3 | 1.049370e-1 | 1,535 |
| Balanced | 4,061 | 29.879 | 4.231545e-3 | 1.052954e-1 | 1,557 |
| Moderate | 3,460 | 25.373 | 5.655367e-3 | 1.113168e-1 | 1,540 |
| Aggressive | 2,281 | 18.867 | 9.300376e-3 | 1.582153e-1 | 1,834 |

The aggressive setting is rejected because both error norms and the DoF count
degrade substantially. Moderate is viable when L2 accuracy is the priority,
but its H1 error is about 6% higher. Balanced preserves H1 accuracy and gives
the best overall tradeoff.

## Balanced validation over all timesteps

| dt | Strict training (s) | Balanced training (s) | Training reduction | Balanced L2 change | Balanced H1 change |
|---:|---:|---:|---:|---:|---:|
| 0.0100 | 27.296 | 22.245 | 18.5% | -6.2% | -4.7% |
| 0.0050 | 29.141 | 26.685 | 8.4% | -14.7% | -1.7% |
| 0.0025 | 31.940 | 29.879 | 6.5% | -28.6% | +0.3% |

Negative error changes mean improved accuracy. The balanced configuration
reduces neural-training time at every timestep. It improves L2 accuracy in all
three cases, improves H1 accuracy for two cases, and changes the finest-step
H1 result by only +0.3%.

Total elapsed time also fell from 46.85 to 39.74 seconds at `dt=0.01`, from
52.85 to 49.35 seconds at `dt=0.005`, and from 67.75 to 63.97 seconds at
`dt=0.0025`. These are wall-clock measurements and remain sensitive to system
load, while the deterministic error results are reproducible.

## Recommendation

Use the **balanced** L-BFGS configuration with the width-35, three-hidden-layer
network:

```cpp
const unsigned int iterations = is_initial_fit ? 2100 : 90;
options.max_iter(iterations);
options.max_eval(iterations * 5 / 4);
options.tolerance_grad(1e-8);
options.tolerance_change(1e-11);
options.history_size(75);
options.line_search_fn("strong_wolfe");
```

In the implementation, the initial/later budgets are selected in
`transfer-strategy-comparison.cc`; the other options are set in
`neural_surrogate.cc`.
