#pragma once

#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <vector>

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

struct TrainingResult
{
  unsigned int closure_evaluations = 0;
  double final_loss = 0.0;
  double seconds = 0.0;
};

class NeuralSurrogate
{
public:
  NeuralSurrogate();

  TrainingResult train(const std::vector<std::array<double, 2>> &points,
                       const std::vector<double> &values,
                       const unsigned int max_iterations);
  std::vector<double>
  values(const std::vector<std::array<double, 2>> &points);
  double value(const std::array<double, 2> &point);
  bool is_trained() const;

private:
  SurrogateNet network;
  bool trained = false;
};
