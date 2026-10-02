# Adaptive rotating Gaussian with deal.II and LibTorch

This project solves the two-dimensional rotating-Gaussian parabolic problem
with deal.II 9.7.1 and Q1 quadrilateral finite elements. It also contains a
small LibTorch executable used to verify the C++ neural-network dependency
before it is connected to the finite-element solver.

## Repository map

Start with the folder that matches the question you want to study:

| Directory | Purpose |
|---|---|
| `rotating-gaussian-classical/` | Classical Q1 adaptive FEM baseline, Step-26-style solution transfer, and refinement-fraction studies. |
| `rotating-gaussian-nn-prototype/` | Earlier NN-only transfer prototype plus the paper authors' reference source. Retained for implementation history. |
| `rotating-gaussian-benchmark/` | Recommended Figure 8 project. Runs both classical and neural transfer and produces matched CSV, VTU/PVD, and comparison plots. |
| `rotating-gaussian-transfer-strategy-comparison/` | Three-way study of classical transfer, one NN fit per time level, and repeated NN fitting at every mesh crossing. |
| `mixed-afem-nn/` | Separate mixed finite-element neural-transfer experiments. |
| `step-26/` | Local deal.II Step-26 reference material. |

For current rotating-Gaussian results, use
`rotating-gaussian-benchmark/README.md`. The similarly named historical
directories have been renamed so their roles are explicit.

For the supervisor-requested training-frequency experiment, use
`rotating-gaussian-transfer-strategy-comparison/README.md`. Its
`nn-once-per-time` mode is the recommended strategy and also documents the
actual behavior of the current Figure 8 benchmark.

The recommended Windows development environment is **WSL 2 with Ubuntu**.
Although VS Code runs as a Windows application, the compiler, CMake,
deal.II, LibTorch, source files, and executable all remain inside Ubuntu.

## 1. Windows setup: WSL, Ubuntu, deal.II, and VS Code

### 1.1 Install WSL 2 and Ubuntu

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested. Launch Ubuntu from the Start menu and create the
Linux username and password requested on first launch. The password is used by
`sudo`; Linux does not display characters while it is being entered.

Confirm that Ubuntu uses WSL 2:

```powershell
wsl --list --verbose
```

If necessary, convert it:

```powershell
wsl --set-version Ubuntu 2
```

Reference: [Microsoft's WSL installation guide](https://learn.microsoft.com/windows/wsl/install).

### 1.2 Install deal.II and the build tools inside Ubuntu

The following commands belong in the **Ubuntu terminal**, not PowerShell:

```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential cmake ninja-build git curl libdeal.ii-dev
```

Optionally install the deal.II examples and documentation:

```bash
sudo apt install libdeal.ii-doc
```

Check the installed package version:

```bash
dpkg-query -W -f='${Version}\n' libdeal.ii-dev
```

The Ubuntu release determines which deal.II version `apt` provides. This
project requests deal.II 9.7.1 in `CMakeLists.txt`; if the command above shows
an older release, use a newer supported Ubuntu release or build the required
deal.II release from source. The official source-build outline is available in
the [deal.II repository](https://github.com/dealii/dealii).

The deal.II project also documents the Ubuntu package names on its
[Windows/WSL page](https://github.com/dealii/dealii/wiki/Windows).

### 1.3 Keep the project in the Linux filesystem

For better compilation and filesystem performance, put C++ projects under the
WSL home directory rather than `/mnt/c`:

```bash
mkdir -p ~/dealII-projects
cd ~/dealII-projects
```

For this project, the expected directory is:

```text
/home/<linux-user>/dealII-projects/rotating-gaussian-classical
```

Windows can browse it at:

```text
\\wsl.localhost\Ubuntu\home\<linux-user>\dealII-projects\rotating-gaussian-classical
```

Do not compile the same build directory from both Windows CMake and Ubuntu
CMake. This project is compiled entirely inside Ubuntu.

### 1.4 Install and connect VS Code

1. Install [Visual Studio Code](https://code.visualstudio.com/) on Windows.
2. In VS Code, install the Microsoft **WSL** extension
   (`ms-vscode-remote.remote-wsl`).
3. In the Ubuntu terminal, open the project:

   ```bash
   cd ~/dealII-projects/rotating-gaussian-classical
   code .
   ```

4. Wait while VS Code installs its small server inside WSL.
5. Confirm that the bottom-left status indicator says **WSL: Ubuntu**.
6. Install the Microsoft **C/C++** and **CMake Tools** extensions in the WSL
   extension host when VS Code prompts for them.

All VS Code terminals and build commands in this window now execute inside
Ubuntu. See the official [VS Code WSL tutorial](https://code.visualstudio.com/docs/remote/wsl-tutorial).

## 2. Native Ubuntu setup

On a computer running Ubuntu directly, WSL and the VS Code WSL extension are
not needed. Install the toolchain and deal.II with:

```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential cmake ninja-build git curl libdeal.ii-dev
```

Optional packages:

```bash
sudo apt install libdeal.ii-doc paraview
```

Verify the tools:

```bash
c++ --version
cmake --version
dpkg-query -W -f='${Version}\n' libdeal.ii-dev
```

Configure and build the FEM solver from the project root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fig8-quads -j2
```

Run it from the build directory so its VTU and PVD files are written there:

```bash
cd build
./fig8-quads
```

Open `rotating-gaussian.pvd` in ParaView to load the entire time series. Each
`rotating-gaussian-XXXX.vtu` file is one individual mesh/solution snapshot.

## 3. LibTorch setup

### 3.1 What CMake does

CMake is a C++ build-configuration system, not a package installer like
Python's `pip`. LibTorch must first be downloaded and extracted. CMake then
reads `TorchConfig.cmake` to obtain its include paths, ABI flags, libraries,
and runtime paths.

The official procedure is documented in
[Installing C++ Distributions of PyTorch](https://docs.pytorch.org/cppdocs/installing.html).

### 3.2 Download the CPU C++ distribution

This project has been tested with **LibTorch 2.7.1, CPU, shared libraries,
with dependencies, C++11 ABI**. From the project root:

```bash
mkdir -p third_party
cd third_party
curl -L --fail -o libtorch-2.7.1-cpu.zip \
  'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip'
cmake -E tar xf libtorch-2.7.1-cpu.zip
cd ..
```

Using `cmake -E tar` avoids requiring the separate `unzip` package. Verify the
installation:

```bash
cat third_party/libtorch/build-version
test -f third_party/libtorch/share/cmake/Torch/TorchConfig.cmake
```

Expected version output:

```text
2.7.1+cpu
```

For a different release or a CUDA build, select a matching package on the
[PyTorch website](https://pytorch.org/). Do not mix a CUDA LibTorch package
with an incompatible NVIDIA driver/CUDA environment. The C++11-ABI builds
require a sufficiently recent GCC and glibc; the current official requirements
are listed in the PyTorch installation guide.

### 3.3 CMake integration used by this project

This section changes exactly one file: **`CMakeLists.txt` in the root of your
C++ project**. Do not enter the following CMake commands in the Ubuntu
terminal, and do not paste them into a `.cc` or `.h` file.

For example, if the project is stored at:

```text
/home/your-username/dealII-projects/my-project/
```

then edit:

```text
/home/your-username/dealII-projects/my-project/CMakeLists.txt
```

Before continuing, the directory should look approximately like this:

```text
my-project/
|-- CMakeLists.txt             <-- edit this file
|-- main.cc                    <-- your deal.II program
|-- neural_surrogate.cc        <-- only if your program uses this file
|-- neural_surrogate.h         <-- only if your program uses this file
`-- third_party/
    `-- libtorch/
        |-- include/
        |-- lib/
        `-- share/cmake/Torch/TorchConfig.cmake
```

From the project root, confirm that the required LibTorch configuration file
exists:

```bash
test -f third_party/libtorch/share/cmake/Torch/TorchConfig.cmake \
  && echo "TorchConfig.cmake found"
```

#### 3.3.1 Adding LibTorch to an existing deal.II executable

Open the existing `CMakeLists.txt`. It will already contain an
`add_executable(...)` command for the deal.II program. Use the **same target
name** in every later target command.

The complete file can follow this pattern:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_dealii_project LANGUAGES CXX)

# Find deal.II first.
find_package(deal.II 9.7.1 REQUIRED
  HINTS /usr/lib/x86_64-linux-gnu)

# Find the LibTorch folder stored inside this project.
find_package(Torch REQUIRED
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
  NO_DEFAULT_PATH)

deal_ii_initialize_cached_variables()

# Replace my-program and the source filenames with your real names.
add_executable(my-program
  main.cc
  neural_surrogate.cc)

# Configure this executable for deal.II.
deal_ii_setup_target(my-program)

# Link and compile this same executable with LibTorch.
target_link_libraries(my-program PRIVATE ${TORCH_LIBRARIES})
target_compile_options(my-program PRIVATE ${TORCH_CXX_FLAGS})

set_property(TARGET my-program PROPERTY CXX_STANDARD 17)
set_property(TARGET my-program PROPERTY CXX_STANDARD_REQUIRED ON)
```

Make these substitutions:

1. Replace every occurrence of `my-program` with the executable target name
   from your `add_executable(...)` command.
2. Replace `main.cc` with the name of your main C++ source file.
3. Keep `neural_surrogate.cc` only if that file exists and is part of your
   program. List every required `.cc` file inside `add_executable(...)`.

For example, if the program is declared with:

```cmake
add_executable(heat-solver heat.cc neural_surrogate.cc)
```

then all commands must use the target name `heat-solver`:

```cmake
deal_ii_setup_target(heat-solver)
target_link_libraries(heat-solver PRIVATE ${TORCH_LIBRARIES})
target_compile_options(heat-solver PRIVATE ${TORCH_CXX_FLAGS})
set_property(TARGET heat-solver PROPERTY CXX_STANDARD 17)
set_property(TARGET heat-solver PROPERTY CXX_STANDARD_REQUIRED ON)
```

Do not use `torch-smoke-test` in these commands unless that is genuinely the
name of the executable being configured.

#### 3.3.2 Optional separate smoke-test executable

The repository also contains `torch-smoke-test.cc`. This small program checks
LibTorch without changing the FEM solver. To build it, add a **second target**
to the same root `CMakeLists.txt`, after `find_package(Torch ...)`:

```cmake
add_executable(torch-smoke-test torch-smoke-test.cc)
target_link_libraries(torch-smoke-test PRIVATE ${TORCH_LIBRARIES})
target_compile_options(torch-smoke-test PRIVATE ${TORCH_CXX_FLAGS})
set_property(TARGET torch-smoke-test PROPERTY CXX_STANDARD 17)
set_property(TARGET torch-smoke-test PROPERTY CXX_STANDARD_REQUIRED ON)
```

This block is optional. It requires `torch-smoke-test.cc` to be located beside
`CMakeLists.txt`. It does not replace the `add_executable(...)` block for the
actual deal.II solver.

#### 3.3.3 Final check before running CMake

Confirm all four points:

- `CMakeLists.txt` is in the project root.
- `third_party/libtorch/share/cmake/Torch/TorchConfig.cmake` exists.
- Every `.cc` file listed by `add_executable(...)` exists in the project.
- The target name is spelled identically in `add_executable`,
  `deal_ii_setup_target`, `target_link_libraries`, `target_compile_options`,
  and `set_property`.

### 3.4 Build and run the LibTorch smoke test

Use a separate build directory:

```bash
cmake -S . -B build-libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build build-libtorch --target torch-smoke-test -j1
./build-libtorch/torch-smoke-test
```

Use `-j1` initially because compiling `torch/torch.h` can require substantial
memory. A successful run ends with:

```text
LibTorch smoke test passed.
```

Build the deal.II solver using the same configuration:

```bash
cmake --build build-libtorch --target fig8-quads -j1
```

Inspect the dynamically linked Torch libraries if troubleshooting:

```bash
ldd build-libtorch/torch-smoke-test | grep -E 'libtorch|libc10'
```

### 3.5 Common problems

**CMake cannot find Torch**

Confirm that this file exists:

```text
third_party/libtorch/share/cmake/Torch/TorchConfig.cmake
```

If LibTorch is installed elsewhere, either update the `PATHS` entry in
`CMakeLists.txt` or configure with:

```bash
cmake -S . -B build-libtorch \
  -DTorch_DIR=/absolute/path/to/libtorch/share/cmake/Torch
```

**Compilation runs out of memory**

Build the Torch target with one job:

```bash
cmake --build build-libtorch --target torch-smoke-test -j1
```

**Undefined C++ symbols while linking**

Use the C++11-ABI LibTorch package and preserve `${TORCH_CXX_FLAGS}` in the
target's compile options. Do not combine object files compiled with different
`_GLIBCXX_USE_CXX11_ABI` values.

**The executable cannot find `libtorch.so` or `libc10.so`**

Reconfigure from a clean build directory so CMake records the correct runtime
path. As a diagnostic fallback:

```bash
export LD_LIBRARY_PATH="$PWD/third_party/libtorch/lib:${LD_LIBRARY_PATH}"
```

## 4. Verified project commands

The following sequence builds both components after deal.II and LibTorch are
installed:

```bash
cd ~/dealII-projects/rotating-gaussian-classical
cmake -S . -B build-libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build build-libtorch --target torch-smoke-test -j1
./build-libtorch/torch-smoke-test
cmake --build build-libtorch --target fig8-quads -j1
```

At the time this README was written, the verified environment was:

- deal.II 9.7.1
- LibTorch 2.7.1+cpu, C++11 ABI
- GCC 15.2.0
- CMake 4.2.3
- Ubuntu under WSL 2
