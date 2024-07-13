# ========================================================================================================= #
#    MOTIVATION: If we can normalize the input features to make optimization run faster,                    #
#    can we normalize the activations (i.e., the inputs to the next layer) to make optimization run faster? #
#    BatchNorm is to normalize the z -- the pre-activation.                                                 #
# ========================================================================================================= #

"""
Last edited on: Jul 5, 2024
by: Lam Thai Nguyen
"""

import torch
import torch.nn as nn


torch.manual_seed(42)

# Input
X = torch.tensor([[1.0, 1.5, 2.0, 2.5],
                [3.0, 3.5, 4.0, 4.5],
                [5.0, 5.5, 6.0, 6.5]])

# Input properties
m, n = X.size()  # (batch size, num_features)
print(f"input shape: {X.size()}")
print(f"==>> # examples (batch size): {m}")
print(f"==>> # features: {n}\n")

# Pre-activation
pre_activation = nn.Linear(in_features=n, out_features=3, bias=True)
print(f"==>> pre_activation: {pre_activation}\n")
print(f"weight shape: {pre_activation.weight.size()}\n{pre_activation.weight}")
print(f"bias shape: {pre_activation.bias.size()}\n{pre_activation.bias}\n")

# Inference
output = pre_activation(X)
print(f"==>> output (Z): \n{output}\n")

# BatchNorm from scratch
mean = torch.mean(output, dim=0, keepdim=True)  # across the batch dimension
var = torch.var(output, dim=0, unbiased=False, keepdim=True)  # across the batch dimension
eps = 1e-5
gamma = nn.Parameter(torch.ones(output.size()[1]))  # Parameter for scaling, default to 1.
beta = nn.Parameter(torch.zeros(output.size()[1]))  # Parameter for shifting, default to 0.
output_norm_from_scratch = gamma * ((output - mean) / torch.sqrt(var + eps)) + beta
print(f"==>> mean: {mean}")
print(f"==>> var: {var}")
print(f"==>> output_norm_from_scratch:\n{output_norm_from_scratch}\n")

# BatchNorm using nn.BatchNorm1d
batch_norm = nn.BatchNorm1d(num_features=3)  # affine=True -> gamma and beta are learnable parameters
print(f"==>> batch_norm: {batch_norm}\n")

output_norm = batch_norm(output)
print(f"==>> output_norm (Z_tilde):\n{output_norm}")

# Checking the results
assert torch.allclose(output_norm_from_scratch, output_norm, rtol=1e-1), "Outputs are not equal"

"""
Remember:
    Input.size() = 
        (batch_size, num_features) -> nn.BatchNorm1D
        (batch_size, C, H, W) -> nn.BatchNorm2D (used in CNN)
"""