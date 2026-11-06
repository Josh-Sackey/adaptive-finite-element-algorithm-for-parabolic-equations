# Early-time timestep study results

All runs use `T=0.05`, top fraction `0.6`, bottom fraction `0.4`, Q1
quadrilateral elements, backward Euler, and the same spatial/adaptive and NN
settings. The NN is trained once per time level.

| dt | Method | Steps | NN fits | Total time (s) | Final L2 error | Final H1 seminorm | Final DoFs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0100 | Classical | 5 | 0 | 0.866 | 1.292e-3 | 1.858e-1 | 13,684 |
| 0.0100 | NN once/time | 5 | 6 | 28.434 | 3.170e-3 | 9.760e-2 | 1,484 |
| 0.0050 | Classical | 10 | 0 | 1.546 | 1.131e-3 | 1.852e-1 | 13,684 |
| 0.0050 | NN once/time | 10 | 11 | 29.821 | 3.988e-3 | 9.966e-2 | 1,469 |
| 0.0025 | Classical | 20 | 0 | 2.543 | 1.070e-3 | 1.850e-1 | 13,684 |
| 0.0025 | NN once/time | 20 | 21 | 46.223 | 5.825e-3 | 1.054e-1 | 1,502 |

## Interpretation

The classical mesh-adaptation interval was held fixed at `0.05` physical-time
units, giving the same 13,684 DoFs in all three final classical solutions.
Halving the timestep produced modest monotonic L2 improvement at `T=0.05`:
the error fell by about 12.4% and then 5.4%, while the classical H1 error was
nearly unchanged. The NN result moved in the opposite direction: its final L2
error increased by about 26% and then 46% as the timestep was halved. The NN
H1 error also remained near `0.10` instead of improving.

Therefore, backward-Euler timestep error does not explain the NN's poorer
early-time L2 accuracy. The evidence is consistent with NN surrogate error
being injected once per time level and accumulating when a finer timestep
creates more fits. This is an observed correlation, not yet a proof of the
mechanism. A controlled follow-up should separate temporal error from neural
projection error by evaluating the NN transfer against the same accepted mesh
sequence, or by varying the surrogate accuracy while holding `dt` fixed.
