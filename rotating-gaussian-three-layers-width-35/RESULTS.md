# Three-hidden-layer width study

## Goal and controlled setup

The objective is to reduce neural-training cost without materially degrading
the solution accuracy. The baseline `2-40-40-40-1` network was compared with
`2-35-35-35-1` and `2-30-30-30-1`. All other settings were held fixed:

- three `tanh` hidden layers;
- deterministic random seed 42;
- L-BFGS training settings;
- one fit per accepted time level;
- Q1 quadrilateral finite elements and the same adaptive policy;
- `T=0.05`, top fraction 0.6, and bottom fraction 0.4.

The parameter counts are 3,441 (width 40), 2,661 (width 35), and 1,981
(width 30). Width 35 therefore removes 22.7% of the trainable parameters.

## Final errors at T = 0.05

| dt | Width | L2 error | H1 seminorm | DoFs | NN training (s) |
|---:|---:|---:|---:|---:|---:|
| 0.0100 | 40 | 3.169989e-3 | 9.759964e-2 | 1,484 | 27.930 |
| 0.0100 | 35 | 2.708238e-3 | 1.011958e-1 | 1,408 | 27.296 |
| 0.0100 | 30 | 5.304386e-3 | 1.067381e-1 | 1,584 | 24.755 |
| 0.0050 | 40 | 3.987647e-3 | 9.965587e-2 | 1,469 | 29.033 |
| 0.0050 | 35 | 4.377067e-3 | 9.928951e-2 | 1,474 | 29.141 |
| 0.0050 | 30 | 4.256681e-3 | 1.031468e-1 | 1,672 | 27.644 |
| 0.0025 | 40 | 5.824673e-3 | 1.053547e-1 | 1,502 | 44.781 |
| 0.0025 | 35 | 5.929338e-3 | 1.049370e-1 | 1,535 | 31.940 |
| 0.0025 | 30 | 6.045978e-3 | 1.178081e-1 | 1,592 | 30.380 |

## Width-35 accuracy relative to width 40

| dt | L2 change | H1 change | DoF change |
|---:|---:|---:|---:|
| 0.0100 | -14.6% (better) | +3.7% | -5.1% |
| 0.0050 | +9.8% | -0.4% (better) | +0.3% |
| 0.0025 | +1.8% | -0.4% (better) | +2.2% |

Width 30 is faster but does not satisfy the accuracy objective consistently.
Width 35 gives the best observed compromise: it improves one norm in every
case, stays close to width 40 in the other norm, and is especially close at
the finest timestep.

## Same-session timing validation

Wall-clock measurements from different batches were affected by system load.
For a fairer check, width 40 was rerun immediately after width 35 at
`dt=0.0025`. The deterministic endpoint remained exactly the baseline result.
The matched training totals were:

| Width | Training time (s) | L2 error | H1 seminorm |
|---:|---:|---:|---:|
| 40 recheck | 35.756 | 5.824673e-3 | 1.053547e-1 |
| 35 | 31.940 | 5.929338e-3 | 1.049370e-1 |

Under the same current machine conditions, width 35 reduced neural-training
time by **10.7%**, while changing the L2 error by +1.8% and improving the H1
seminorm by 0.4%. This is the strongest candidate tested so far for the stated
speed/accuracy goal.

The complete compact comparison is in
`results/architecture-width-comparison.csv`. Raw VTU/PVD output remains in
the corresponding WSL project directories and is not copied into this
workspace.
