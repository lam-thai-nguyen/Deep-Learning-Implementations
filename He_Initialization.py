# ================================================================================================================================= #
#   He Initialization (also known as Kaiming Initialization)                                                                        #
#                                                                                                                                   #
#   Motivation:         - Random initialization may cause vanishing or exploding gradients, thus make the learning process slow.    #
#   Solution:           - Careful weights initialization can help with the mentioned gradients problems.                            #
#   He Initialization:  - works well with ReLU activation function.                                                                 #
#                       - sets weights = randn((out_features, in_features)) * sqrt(2./in_features)                                  #                                                                                                                           #
#                                                                                                                                   #
#   Related topics: (1) Vanishing, exploding gradients (2) Xavier initialization for sigmoid and tanh                               #
# ================================================================================================================================= #

"""
Last edited on: Jul 19, 2024
by: Lam Thai Nguyen
"""

import torch
import torch.nn as nn
import torch.nn.init as init


seed = 42
torch.manual_seed(seed)

def setup(seed):
    torch.manual_seed(seed)

    # Input
    X = torch.rand((1, 4))
    batch_size, num_features = X.size()

    # Layer
    out_features = 3
    layer1 = nn.Linear(in_features=num_features, out_features=out_features, bias=False)

    return layer1.weight, num_features, out_features
    
# ============== #
#  From Scratch  #
# ============== #
init_weights, num_features, out_features = setup(42)
std_dev = torch.sqrt(torch.tensor(2. / num_features))
with torch.no_grad():
    weights_scratch = torch.randn((out_features, num_features)) * std_dev

# ============== #
#     PyTorch    #
# ============== #
init_weights, num_features, out_features = setup(42)
init.kaiming_normal_(init_weights, mode="fan_in", nonlinearity="relu")

weights_pytorch = init_weights

assert torch.allclose(weights_scratch, weights_pytorch, rtol=1e-5), "Weights are not close!"

# ================================ #
#  Using He Init in a Model class  #
# ================================ #

class Net(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super(Net, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=output_features)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward(self, x):
        pass  # You get the idea
    