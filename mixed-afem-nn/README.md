# Mixed quadrilateral AFEM / neural-transfer benchmark

This program solves the manuscript system

`phi_t - Laplacian(mu) + phi = f`, `-mu - Laplacian(phi) = 0`

with Q1--Q1 continuous elements on adaptive quadrilateral meshes. It compares
deal.II `SolutionTransfer` (`classical`) with neural interpolation of the old
`phi` field (`nn`). The coupled solve recomputes both `phi` and `mu` after every
transfer.

LibTorch may be installed locally at `third_party/libtorch` or shared from
`../rotating-gaussian-classical/third_party/libtorch`, as described in the
repository root README.

```bash
./mixed-afem-nn <example:1|2> <method:classical|nn> <final_time> <dt> <output_dir>
```

Example 2 intentionally follows the current manuscript equation exactly: both
centre coordinates are `0.3*cos(2*pi*t)`. This is diagonal oscillation, not a
circular trajectory. If a circle was intended, the manuscript y-coordinate
should instead use `0.3*sin(2*pi*t)`.
