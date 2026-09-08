# Implementing a LibTorch surrogate in a deal.II time-dependent FEM solver

This guide explains how to add a neural-network surrogate to a deal.II
finite-element program using the working rotating-Gaussian project as a
template. It is written for readers who understand basic C++ and finite
elements but may be new to neural networks or LibTorch.

The template files are:

- `fig8-quads-nn.cc`: finite-element problem, time loop, mesh adaptation, and
  the connection between deal.II and the neural network.
- `neural_surrogate.h`: declarations for the network and wrapper interface.
- `neural_surrogate.cc`: LibTorch network construction, training, and
  inference.
- `CMakeLists.txt`: locates and links deal.II and LibTorch.

## 1. What the neural network does

The neural network does not replace the finite-element PDE solve. At time
`t_(n-1)`, deal.II produces nodal values of the finite-element solution
`u_h^(n-1)`. We train a network to approximate the mapping

```text
(x, y) -> u_h^(n-1)(x, y).
```

The trained function `u_theta^(n-1)(x,y)` can then be evaluated at arbitrary
points, including quadrature points on a completely new mesh.

For backward Euler, the discrete equation is

```text
(M + dt A) u_h^n = M u_theta^(n-1) + dt F^n.
```

The mass matrix `M`, stiffness matrix `A`, load vector `F^n`, and unknown
`u_h^n` still come from deal.II. Only the previous-time function on the
right-hand side is supplied by the neural network.

This provides a continuous, mesh-independent representation of the previous
solution. The program can discard the old mesh, construct a fresh adaptive
mesh, and still evaluate the old solution wherever it is needed.

## 2. Why the architecture is `[2,40,40,40,1]`

The notation lists the number of neurons in each layer:

```text
2 inputs -> 40 -> 40 -> 40 -> 1 output
```

### Two input neurons

The surrogate represents a scalar field in two spatial dimensions. Each
training sample supplies the coordinate pair

```text
x_input = [x, y].
```

Therefore, the input layer has two neurons. For a three-dimensional problem,
this would become three inputs `[x,y,z]`. Time is not an input in this design
because a separate surrogate state is trained at every time level. A single
space-time network would instead require three inputs `[x,y,t]`.

### Three hidden layers of 40 neurons

The hidden layers provide the nonlinear capacity needed to represent a narrow,
smooth Gaussian. Forty neurons is not derived from the FEM mesh size and is
not equal to the number of degrees of freedom. It is a modeling choice adopted
from the paper's published two-dimensional experiments.

Three layers make the network expressive enough to represent localized curved
features, while 40 neurons per layer keeps training reasonably small. Wider or
deeper networks may fit more complicated solutions but require more memory and
optimization work. Smaller networks train faster but can underfit sharp peaks.

The architecture should therefore be treated as a tested starting point, not
as a universal optimum. For a new PDE, compare training loss and FEM error for
several architectures before changing it.

### One output neuron

The PDE has one scalar unknown `u`, so the network returns one number:

```text
u_prediction = u_theta(x,y).
```

A vector-valued PDE would normally need one output for each solution
component.

## 3. Declare the network in `neural_surrogate.h`

Import the LibTorch C++ API:

```cpp
#include <torch/torch.h>
```

Declare a module containing four fully connected layers:

```cpp
struct SurrogateNetImpl : torch::nn::Module
{
  SurrogateNetImpl();
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear layer1{nullptr};
  torch::nn::Linear layer2{nullptr};
  torch::nn::Linear layer3{nullptr};
  torch::nn::Linear output{nullptr};
};
TORCH_MODULE(SurrogateNet);
```

`TORCH_MODULE(SurrogateNet)` creates the convenient LibTorch module-holder
type used by the rest of the program.

The wrapper class exposes ordinary C++ containers to the deal.II code:

```cpp
class NeuralSurrogate
{
public:
  TrainingResult train(
    const std::vector<std::array<double, 2>> &points,
    const std::vector<double> &values,
    const unsigned int max_iterations);

  std::vector<double> values(
    const std::vector<std::array<double, 2>> &points);

private:
  SurrogateNet network;
  bool trained = false;
};
```

This wrapper keeps Torch tensor details out of the main FEM class.

## 4. Construct and register the layers

In `neural_surrogate.cc`, construct the architecture:

```cpp
SurrogateNetImpl::SurrogateNetImpl()
  : layer1(register_module("layer1", torch::nn::Linear(2, 40)))
  , layer2(register_module("layer2", torch::nn::Linear(40, 40)))
  , layer3(register_module("layer3", torch::nn::Linear(40, 40)))
  , output(register_module("output", torch::nn::Linear(40, 1)))
{}
```

Every trainable layer must be passed through `register_module`. Registration
allows LibTorch to find the weights and biases when
`network->parameters()` is given to the optimizer. An unregistered layer will
not be trained correctly.

## 5. Initialize parameters and select precision

The template initializes weights with Kaiming uniform initialization and sets
biases to zero:

```cpp
torch::nn::init::kaiming_uniform_(linear->weight,
                                 std::sqrt(5.0),
                                 torch::kFanIn,
                                 torch::kTanh);
torch::nn::init::zeros_(linear->bias);
```

Initialization prevents every neuron from starting with identical parameters
and provides a numerically reasonable scale for optimization.

The wrapper sets a reproducible random seed and uses double precision:

```cpp
torch::manual_seed(42);
network->to(torch::kFloat64);
```

deal.II vectors in this project use `double`. Keeping the network, input
tensors, and target tensors in `torch::kFloat64` avoids float/double type
mismatches and preserves numerical accuracy.

## 6. Define the forward calculation

The forward method applies a linear transformation and `tanh` activation at
each hidden layer:

```cpp
torch::Tensor SurrogateNetImpl::forward(torch::Tensor x)
{
  x = torch::tanh(layer1(x));
  x = torch::tanh(layer2(x));
  x = torch::tanh(layer3(x));
  return output(x);
}
```

`tanh` is smooth, which is helpful when approximating a smooth PDE solution.
The output layer is linear: the solution is a real-valued quantity and should
not be artificially restricted to a classification range.

The current implementation does not force positivity or impose Dirichlet
conditions inside the neural-network formula. Exact Dirichlet values are still
imposed by the FEM system. If a future application needs an exactly
boundary-conforming surrogate, use a construction such as
`u_theta = g_tilde + d(x) N_theta(x)`, where `d=0` on the boundary.

## 7. Extract training data from deal.II

After solving the FEM problem, `train_surrogate()` obtains the physical
coordinate of every Q1 degree of freedom:

```cpp
std::map<types::global_dof_index, Point<2>> support_points;
DoFTools::map_dofs_to_support_points(MappingQ1<2>(),
                                     dof_handler,
                                     support_points);
```

It then builds one coordinate and one target value for each DoF:

```cpp
std::vector<std::array<double, 2>> points(dof_handler.n_dofs());
std::vector<double> values(dof_handler.n_dofs());

for (const auto &entry : support_points)
{
  points[entry.first] = {entry.second[0], entry.second[1]};
  values[entry.first] = solution[entry.first];
}
```

For Q1 scalar elements, a DoF is associated with a vertex, so the training set
is effectively

```text
(vertex x, vertex y) -> FEM nodal solution value.
```

For higher-order elements, support points also occur inside cells and on
edges. The same concept works, provided the finite element has well-defined
support points.

## 8. Convert C++ data to tensors

`NeuralSurrogate::train()` creates tensors with shapes `N x 2` and `N x 1`:

```cpp
auto x = torch::empty({static_cast<long>(points.size()), 2},
                      torch::kFloat64);
auto y = torch::empty({static_cast<long>(values.size()), 1},
                      torch::kFloat64);
```

The first tensor contains coordinates and the second contains FEM targets.
Accessors copy the C++ values into the tensors:

```cpp
xa[i][0] = points[i][0];
xa[i][1] = points[i][1];
ya[i][0] = values[i];
```

Always verify both tensor shapes and dtypes. Common LibTorch errors come from
passing `float64` data into a `float32` network or reversing the sample and
feature dimensions.

## 9. Define the loss and train with L-BFGS

The training objective is mean squared error:

```cpp
const auto prediction = network->forward(x);
const auto loss = torch::mse_loss(prediction, y);
```

Mathematically,

```text
loss(theta) = (1/N) sum_i |u_theta(x_i,y_i) - u_h(x_i,y_i)|^2.
```

The optimizer is configured as follows:

```cpp
torch::optim::LBFGSOptions options(0.5);
options.max_iter(max_iterations);
options.max_eval(max_iterations * 5 / 4);
options.tolerance_grad(1e-9);
options.tolerance_change(1e-12);
options.history_size(100);
options.line_search_fn("strong_wolfe");

torch::optim::LBFGS optimizer(network->parameters(), options);
```

L-BFGS repeatedly invokes a closure. The closure clears old gradients,
calculates predictions and loss, performs automatic differentiation, and
returns the loss:

```cpp
auto closure = [&]() {
  optimizer.zero_grad();
  const auto prediction = network->forward(x);
  const auto loss = torch::mse_loss(prediction, y);
  loss.backward();
  return loss;
};

optimizer.step(closure);
```

Calling `zero_grad()` is essential. Otherwise, gradients accumulate across
closure evaluations and the optimizer receives incorrect derivatives.

The template allows up to 2,500 iterations for the first fit and 100 for later
fits:

```cpp
const unsigned int iterations = surrogate.is_trained() ? 100 : 2500;
```

The later fits are warm starts. The network object is retained, so training at
time `t_n` begins with the weights learned at `t_(n-1)`. Because `dt=0.01`,
consecutive solutions are similar and normally require much less training.

## 10. Evaluate the surrogate on a new FEM mesh

Inference does not need parameter gradients, so it uses:

```cpp
torch::NoGradGuard no_grad;
network->eval();
auto prediction = network->forward(x).contiguous();
```

`NoGradGuard` reduces memory and computation. The predicted tensor is copied
back into `std::vector<double>` for the deal.II assembly routine.

At every quadrature point on every active cell, the main solver collects the
new physical coordinates:

```cpp
points[q] = {fe_values.quadrature_point(q)[0],
             fe_values.quadrature_point(q)[1]};
previous_values = surrogate.values(points);
```

The predicted old solution enters the local backward-Euler load vector:

```cpp
cell_rhs(i) += fe_values.shape_value(i, q) *
               (previous_values[q] +
                time_step * rhs.value(fe_values.quadrature_point(q))) *
               fe_values.JxW(q);
```

This corresponds to integrating

```text
phi_i (u_theta^(n-1) + dt f^n)
```

over each cell.

## 11. Position training in the time loop

The required order is:

```text
1. Solve the initial FEM problem.
2. Train u_theta^0 from the initial FEM solution.
3. Advance to t_n.
4. Build or adapt the new mesh.
5. Evaluate u_theta^(n-1) during FEM assembly.
6. Solve for u_h^n.
7. Train the retained network on u_h^n.
8. Repeat.
```

In the template, the old mesh is explicitly discarded:

```cpp
dof_handler.clear();
triangulation.clear();
```

The program recreates the coarse mesh and adaptively solves on it. This is the
key reason for the surrogate: the old FEM vector cannot simply be indexed on
an unrelated new mesh, but the network can be evaluated at any coordinate.

## 12. Link LibTorch with CMake

The project finds deal.II and the local LibTorch distribution:

```cmake
find_package(deal.II 9.7.1 REQUIRED
  HINTS /usr/lib/x86_64-linux-gnu)

find_package(Torch REQUIRED
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
  NO_DEFAULT_PATH)
```

The executable contains both the FEM and surrogate source files:

```cmake
add_executable(fig8-quads-nn
  fig8-quads-nn.cc
  neural_surrogate.cc)
```

Finally, apply deal.II's target configuration and link Torch:

```cmake
deal_ii_setup_target(fig8-quads-nn)
target_link_libraries(fig8-quads-nn ${TORCH_LIBRARIES})
target_compile_options(fig8-quads-nn PRIVATE ${TORCH_CXX_FLAGS})
```

CMake does not install LibTorch like Python's `pip`. It locates the already
downloaded C++ package, supplies include paths and compiler flags, and links
the executable against the required Torch libraries.

## 13. Build and run the template

From the project directory:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/third_party/libtorch"
cmake --build build -j2
```

Run from a dedicated output directory:

```bash
mkdir -p runs/my-run
cd runs/my-run
../../build/fig8-quads-nn 0.6 1.0
```

The arguments are the top refinement fraction and final time. The bottom
fraction is set to `1-top_fraction`.

## 14. Verify that the network is working

Do not validate the neural network only by checking that the executable runs.
Check all of the following:

1. `neural-training.csv` contains one row for `t=0` and every later time.
2. `final_mse` decreases to an acceptable level. The verified full run ended
   with an MSE of approximately `1.28e-5`.
3. The initial fit is more expensive than warm-start fits.
4. `simulation-history.csv` shows bounded FEM errors through the full orbit.
5. `solution` and `exact_solution` agree visually in ParaView.
6. The Gaussian returns close to its initial location at `t=1`.
7. Temporarily replacing neural predictions with the exact previous solution
   gives a useful reference for separating NN error from FEM error.

The verified Figure 8-aligned run ended with approximately:

```text
L2 error          = 2.015e-3
H1 seminorm error = 1.140e-1
NN MSE            = 1.283e-5
```

## 15. Adapting the template to another PDE

For a new scalar two-dimensional parabolic problem:

1. Keep two inputs and one output.
2. Replace the exact solution, forcing term, and boundary data in the FEM
   program.
3. Continue training on `(support point, FEM value)` pairs.
4. Evaluate the surrogate at the quadrature points used by the time-discrete
   weak form.
5. Choose the initial and warm-start training budgets from measured loss, not
   by assumption.
6. Recheck whether coordinates should be normalized to a convenient range.
7. Compare against a conventional `SolutionTransfer` run.

For systems, 3D domains, or space-time networks, change the input/output layer
sizes deliberately:

| Problem | Suggested input size | Suggested output size |
|---|---:|---:|
| Scalar PDE in 2D, separate network per time | 2 | 1 |
| Scalar PDE in 3D, separate network per time | 3 | 1 |
| Scalar space-time model in 2D | 3 (`x,y,t`) | 1 |
| Two-component PDE in 2D | 2 | 2 |

The hidden width and depth remain tunable hyperparameters.

## 16. Important limitations of this template

- The network approximates nodal FEM data, so it also learns FEM discretization
  error.
- Mean squared error weights every training node equally. Adaptive meshes
  contain many nodes near refined features, so the data distribution is not
  spatially uniform.
- The surrogate does not exactly enforce positivity or boundary conditions.
- A low training MSE does not by itself guarantee a low PDE error.
- The current quadrilateral refinement is an analogue of the paper's method;
  it is not the paper's triangular target-mesh generator.
- The current adaptive loop reaches its seven-iteration guard before the
  relative estimator reaches `0.01`, so the mesh-generation component still
  needs improvement.

Use the neural network as a numerical transfer mechanism whose accuracy must
be monitored together with the FEM error—not as a replacement for the PDE,
weak formulation, or a posteriori error estimator.
