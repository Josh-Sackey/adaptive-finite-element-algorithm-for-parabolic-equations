#pragma once

#include <torch/torch.h>

#include <array>
#include <vector>

struct SurrogateNetImpl : torch::nn::Module
{
  SurrogateNetImpl();
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, output{nullptr};
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
                       unsigned int max_iterations);
  std::vector<double> values(const std::vector<std::array<double, 2>> &points);
  bool is_trained() const { return trained; }

private:
  SurrogateNet network;
  bool trained = false;
  double value_mean = 0.0;
  double value_scale = 1.0;
};
