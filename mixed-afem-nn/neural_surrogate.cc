#include "neural_surrogate.h"

#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

SurrogateNetImpl::SurrogateNetImpl()
  : layer1(register_module("layer1", torch::nn::Linear(2, 40)))
  , layer2(register_module("layer2", torch::nn::Linear(40, 40)))
  , layer3(register_module("layer3", torch::nn::Linear(40, 40)))
  , output(register_module("output", torch::nn::Linear(40, 1)))
{}

torch::Tensor SurrogateNetImpl::forward(torch::Tensor x)
{
  x = torch::tanh(layer1(x));
  x = torch::tanh(layer2(x));
  x = torch::tanh(layer3(x));
  return output(x);
}

NeuralSurrogate::NeuralSurrogate()
  : network(SurrogateNet())
{
  torch::manual_seed(42);
  network->to(torch::kFloat64);
}

TrainingResult NeuralSurrogate::train(
  const std::vector<std::array<double, 2>> &points,
  const std::vector<double> &values,
  const unsigned int max_iterations)
{
  if (points.empty() || points.size() != values.size())
    throw std::runtime_error("Invalid neural-surrogate training data");

  value_mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double variance = 0.0;
  for (const double v : values)
    variance += (v - value_mean) * (v - value_mean);
  value_scale = std::sqrt(variance / values.size());
  if (value_scale < 1e-14)
    value_scale = 1.0;

  auto x = torch::empty({static_cast<long>(points.size()), 2}, torch::kFloat64);
  auto y = torch::empty({static_cast<long>(values.size()), 1}, torch::kFloat64);
  auto xa = x.accessor<double, 2>();
  auto ya = y.accessor<double, 2>();
  for (std::size_t i = 0; i < points.size(); ++i)
    {
      xa[i][0] = points[i][0];
      xa[i][1] = points[i][1];
      ya[i][0] = (values[i] - value_mean) / value_scale;
    }

  torch::optim::LBFGSOptions options(0.5);
  options.max_iter(max_iterations);
  options.max_eval(max_iterations * 5 / 4);
  options.tolerance_grad(1e-9);
  options.tolerance_change(1e-12);
  options.history_size(100);
  options.line_search_fn("strong_wolfe");
  torch::optim::LBFGS optimizer(network->parameters(), options);

  network->train();
  TrainingResult result;
  const auto start = std::chrono::steady_clock::now();
  auto closure = [&]() {
    optimizer.zero_grad();
    const auto prediction = network->forward(x);
    const auto loss = torch::mse_loss(prediction, y);
    loss.backward();
    ++result.closure_evaluations;
    result.final_loss = loss.item<double>();
    return loss;
  };
  optimizer.step(closure);
  result.seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  trained = true;
  return result;
}

std::vector<double>
NeuralSurrogate::values(const std::vector<std::array<double, 2>> &points)
{
  if (!trained)
    throw std::runtime_error("Neural surrogate used before training");
  torch::NoGradGuard no_grad;
  auto x = torch::empty({static_cast<long>(points.size()), 2}, torch::kFloat64);
  auto xa = x.accessor<double, 2>();
  for (std::size_t i = 0; i < points.size(); ++i)
    {
      xa[i][0] = points[i][0];
      xa[i][1] = points[i][1];
    }
  network->eval();
  auto prediction = network->forward(x).contiguous();
  const auto pa = prediction.accessor<double, 2>();
  std::vector<double> result(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    result[i] = value_mean + value_scale * pa[i][0];
  return result;
}
