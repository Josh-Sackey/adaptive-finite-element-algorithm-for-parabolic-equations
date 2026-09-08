#include <torch/torch.h>

#include <iostream>

int main()
{
  torch::manual_seed(42);

  const torch::Tensor x = torch::randn({3, 2});
  const torch::Tensor weights = torch::tensor({{2.0}, {-1.0}});
  const torch::Tensor y = torch::relu(torch::matmul(x, weights));

  std::cout << "LibTorch version: " << TORCH_VERSION << '\n'
            << "Input:\n" << x << '\n'
            << "ReLU(xW):\n" << y << '\n'
            << "LibTorch smoke test passed." << std::endl;

  return 0;
}
