# Easy Docker and Apptainer guide

This guide is written for a colleague who does not need to know deal.II,
LibTorch, CMake, or the internal numerical implementation.

## What this package runs

The program solves Example 4.7 from the neural-network-enhanced adaptive finite
element paper: the two-dimensional Allen-Cahn equation on `[-1,1]^2`. It uses
quadrilateral finite elements, Newton's method, adaptive mesh refinement, and a
`2-40-40-40-1` LibTorch neural network for solution transfer.

There are five named run choices:

| Choice | Purpose | Expected workload |
|---|---|---|
| `smoke` | Two-step installation check | A few minutes or less |
| `baseline` | Reproduce our completed paper-aligned run | About 3 minutes on Joshua's laptop |
| `spatial-fine` | Finer mesh and tighter estimator tolerance | Hours; machine dependent |
| `time-fine` | Half-sized time step, 200 steps | Roughly twice baseline or more |
| `production` | Fine mesh plus half-sized time step | Primary supercomputer run; allow up to 12 hours initially |

The most scientifically useful supercomputer task is the **four-case convergence
campaign**: baseline, spatial-fine, time-fine, and production. It separates
spatial error from temporal error and tells us whether the reported interface,
energy, overshoot, and neural-transfer behavior are converging. Merely running
past `T=0.03` would study a different question and is not the first priority.

## What your colleague needs

For Docker on an ordinary Linux workstation:

- Docker installed and running.
- At least 40 GB free disk space for the first build.
- Internet access during the first build.

For the university cluster:

- Apptainer (or Singularity) available.
- Permission to run a supplied `.sif` file.
- One standard 64-bit Intel/AMD Linux compute node.
- Recommended initial request: 8 CPU cores, 32 GB RAM, and 12 hours.

The cluster does **not** need deal.II or LibTorch installed. They are inside the
container. No NVIDIA GPU is required.

## Files to give the colleague

The easiest handoff is:

```text
allen-cahn-newton-nn.sif
scripts/run-slurm.sh       # if the cluster uses Slurm
scripts/run-pbs.sh         # if the cluster uses PBS
docs/HPC-CONTAINER-GUIDE.md
```

The `.sif` file can be large. Use Globus, SCP, SFTP, OneDrive, or the university
file-transfer service instead of email.

## Part A - Build and test with Docker

These steps are for the person preparing the package, not necessarily the
person running it on the cluster.

From the project directory:

```bash
./scripts/build-docker.sh
```

The first build downloads and compiles deal.II 9.7.1 and downloads LibTorch
2.7.1 CPU. It can take 30-90 minutes and uses substantial disk space. Later
builds reuse Docker's cache.

Run the short test:

```bash
./scripts/run-docker.sh smoke
```

Then run the baseline if desired:

```bash
./scripts/run-docker.sh baseline
```

Results appear under:

```text
container-output/smoke/
container-output/baseline/
```

## Part B - Create the Apptainer image

Export the tested Docker image:

```bash
./scripts/export-docker.sh
```

This creates:

```text
allen-cahn-newton-nn.tar
```

On a Linux system with Apptainer installed, convert it:

```bash
./scripts/build-apptainer.sh
```

This creates the portable cluster image:

```text
allen-cahn-newton-nn.sif
```

Apptainer can also build it explicitly from `Apptainer.def`:

```bash
apptainer build allen-cahn-newton-nn.sif Apptainer.def
```

Test the SIF before transferring it:

```bash
./scripts/run-apptainer.sh smoke
```

Create one handoff archive containing the SIF image, scheduler templates,
instructions, and a checksum:

```bash
./scripts/prepare-colleague-package.sh
```

Give the resulting `allen-cahn-colleague-package.tar.gz` file to the colleague.
After extracting it, they can verify that the large image transferred correctly:

```bash
sha256sum -c SHA256SUMS
```

## Part C - Run manually on the cluster

Upload `allen-cahn-newton-nn.sif`, log in to the cluster, and place it in a new
working directory. Then run these commands:

```bash
mkdir -p results
apptainer run \
  --bind "$PWD/results:/results" \
  allen-cahn-newton-nn.sif smoke
```

If the smoke test completes, start the production case:

```bash
apptainer run \
  --bind "$PWD/results:/results" \
  allen-cahn-newton-nn.sif production
```

Do not normally run a long job directly on a login node. Use the scheduler as
shown below.

## Part D - Submit with Slurm

Place these files together:

```text
allen-cahn-newton-nn.sif
scripts/run-slurm.sh
```

From that directory:

```bash
mkdir -p logs results
sbatch scripts/run-slurm.sh
```

The supplied Slurm script launches four array tasks:

1. baseline
2. spatial-fine
3. time-fine
4. production

Check status:

```bash
squeue -u "$USER"
```

Read progress:

```bash
tail -f logs/allen-cahn-JOBID_TASKID.out
```

If the center has different resource rules, staff may edit the `#SBATCH` lines
at the top of `run-slurm.sh`. No C++ changes are needed.

## Part E - Submit with PBS

The supplied PBS template runs the production case:

```bash
mkdir -p results
qsub scripts/run-pbs.sh
```

Check status:

```bash
qstat -u "$USER"
```

## How to know that a run finished correctly

At successful completion, the terminal or scheduler log prints a block like:

```text
==================== RUN COMPLETED ====================
VTU snapshots : 101
Final time    : 0.03
Final DoFs    : ...
Elapsed time  : ... seconds
ParaView file : /results/.../allen-cahn.pvd
Summary CSV   : /results/.../simulation-history.csv
=======================================================
```

Each completed case directory must contain:

```text
allen-cahn.pvd
allen-cahn-0000.vtu, allen-cahn-0001.vtu, ...
simulation-history.csv
newton-history.csv
neural-training.csv
console.log
```

Expected snapshot counts:

| Case | Time steps | Expected VTU files |
|---|---:|---:|
| smoke | 2 | 3 |
| baseline | 100 | 101 |
| spatial-fine | 100 | 101 |
| time-fine | 200 | 201 |
| production | 200 | 201 |

Open `allen-cahn.pvd` in ParaView to view the full animation. Open
`simulation-history.csv` in Excel to inspect time, active cells, degrees of
freedom, estimator, mass, energy, extrema, and elapsed wall time.

## What results should be returned

Ask the colleague to return the entire `results/` directory and the `logs/`
directory. At minimum, preserve all CSV files, `console.log`, and the PVD/VTU
files. The CSV and log files are small; VTU files may be much larger.

For the convergence comparison we will extract:

- final and maximum degrees of freedom;
- total wall time and neural-training time;
- Newton iterations and residuals;
- phase-field minimum and maximum (overshoot);
- mass and discrete energy histories;
- interface position/radius at `t=0`, approximately `0.01`, `0.02`, and `0.03`;
- differences between baseline, spatial-fine, time-fine, and production.

## Common problems

### `apptainer: command not found`

Try the cluster's module system:

```bash
module avail apptainer singularity
module load apptainer
```

If neither name appears, ask cluster support how to run a SIF image.

### Permission denied for a script

Run:

```bash
chmod +x scripts/*.sh
```

### Results disappear after the container exits

The results directory was not bound. Always include:

```bash
--bind "$PWD/results:/results"
```

### Job stops because it exceeds time or memory

The partial logs remain useful. Increase the scheduler request and resubmit only
the failed case. Do not modify the numerical source code.

### The center uses Singularity instead of Apptainer

Replace the command name `apptainer` with `singularity`. The `.sif` format and
run syntax are normally compatible.

## Exact custom-run interface

Advanced users can bypass the presets:

```bash
apptainer run --bind "$PWD/results:/results" allen-cahn-newton-nn.sif \
  custom FINAL_TIME /results/NAME MAX_ADAPT ETA_TOL INITIAL_N MAX_LEVEL INITIAL_ADAPT DT
```

Example equivalent to the baseline:

```bash
apptainer run --bind "$PWD/results:/results" allen-cahn-newton-nn.sif \
  custom 0.03 /results/my-run 7 0.5 8 8 5 0.0003
```

