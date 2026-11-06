# Two-hidden-layer neural-transfer experiment

## Controlled change

The baseline network is `2-40-40-40-1`: two coordinate inputs, three hidden
layers of 40 `tanh` units, and one scalar output. This experiment removes only
the third hidden layer, producing `2-40-40-1`. The random seed, width,
activation function, L-BFGS optimizer, training schedule, finite element
solver, adaptive policy, final time, and timestep values are unchanged.

All results below use `T=0.05`, top fraction `0.6`, bottom fraction `0.4`, Q1
quadrilaterals, backward Euler, and one neural fit per accepted time level.

## Accuracy comparison at T = 0.05

| dt | Hidden layers | Final L2 error | Final H1 seminorm | Final DoFs | NN fits |
|---:|---:|---:|---:|---:|---:|
| 0.0100 | 3 (baseline) | 3.169989e-3 | 9.759964e-2 | 1,484 | 6 |
| 0.0100 | 2 | 4.846771e-3 | 1.174076e-1 | 1,629 | 6 |
| 0.0050 | 3 (baseline) | 3.987647e-3 | 9.965587e-2 | 1,469 | 11 |
| 0.0050 | 2 | 8.472705e-3 | 1.583409e-1 | 1,928 | 11 |
| 0.0025 | 3 (baseline) | 5.824673e-3 | 1.053547e-1 | 1,502 | 21 |
| 0.0025 | 2 | 1.149386e-2 | 2.375527e-1 | 2,113 | 21 |

## Effect of removing one hidden layer

| dt | L2 error change | H1 error change | DoF change |
|---:|---:|---:|---:|
| 0.0100 | +52.9% | +20.3% | +9.8% |
| 0.0050 | +112.5% | +58.9% | +31.2% |
| 0.0025 | +97.3% | +125.5% | +40.7% |

Positive changes mean the two-hidden-layer result is worse or larger than the
three-hidden-layer baseline.

## Timing

The recorded two-layer training totals were 61.10, 16.25, and 20.36 seconds
for `dt=0.01`, `0.005`, and `0.0025`, respectively. These wall-clock values
were affected by machine load: even the unchanged classical controls took
24.8--73.7 seconds in this batch instead of 0.87--2.54 seconds in the earlier
batch. Timing should therefore be repeated under controlled load before using
it to rank the architectures. The error and DoF comparisons are the main
result of this experiment.

## Takeaway

Reducing the network from three hidden layers to two did **not** improve
accuracy. It increased both error norms for every timestep, and the adaptive
algorithm ended with more DoFs. The degradation becomes especially strong for
the finer timesteps, where more neural projections are performed. For this
problem and these fixed training settings, the third hidden layer provides
useful representational capacity for the narrow rotating Gaussian.

The next architecture experiment should keep three hidden layers and vary
either width or training accuracy, rather than adopting the two-layer model as
the new baseline.

Compact machine-readable outputs are in `results/timestep-summary.csv` and
`results/timestep-history.csv`. Raw VTU/PVD simulation output remains in WSL
under `/home/joshua/dealII-projects/rotating-gaussian-two-hidden-layers/runs/`
and is not copied into the repository workspace.
