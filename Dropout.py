# ============================================================================================================================= #
#       What is Dropout?                                                                                                        # 
#           - A regularization technique (i.e., helps with overfitting)                                                         # 
#       How to Dropout?                                                                                                         #
#           - Training phase:                                                                                                   #
#           (1) Goes through each hidden layers in the network.                                                                 #
#           (2) At each hidden layer, randomly select a value for each unit from [0, 1]                                         #
#           (3) Remove any unit whose value is equal or larger than a pre-defined threshold (i.e, keep_prop)                    #
#                   (Removing p=1-keep_prop of all units in a hidden layer)                                                     #
#           - Test time:                                                                                                        #
#           (1) No units are dropped                                                                                            #
#           (2) Scale down weights by multiplying with (1-p)                                                                    #
#       Why Dropout?                                                                                                            #
#           - Makes the network smaller ==>> More computationally efficient                                                     #
#           - Reduce overfitting by not allowing the network to rely heavily on any particular unit                             #
#                                                                                                                               #
#       What is Inverted Dropout?                                                                                               #
#           - A variation of Dropout                                                                                            #
#           - More straightforward and more frequently implemented                                                              #
#       How to Inverted Dropout?                                                                                                #
#           - Training phase:                                                                                                   #
#           (1) Goes through each hidden layers in the network.                                                                 #
#           (2) At each hidden layer, randomly select a value for each unit from [0, 1]                                         #
#           (3) Remove any unit whose value is equal or larger than a pre-defined threshold (i.e, keep_prop)                    #
#                   (Removing p=1-keep_prop of all units in a hidden layer)                                                     #
#           (4) Scale up the activation value by dividing by keep_prop to keep the output's expected value consistent.          #
#           - Test time:                                                                                                        #
#           (1) No units are dropped                                                                                            #
# ============================================================================================================================= #

"""
Last edited on: Jul 13, 2024
by: Lam Thai Nguyen
"""

import torch
import torch.nn as nn


seed = 42
torch.manual_seed(seed)

# Input
X = torch.abs(torch.rand((1, 15))) + 0.1  # (batch_size, num_features)

# A simple network
layer1 = nn.Sequential(
    nn.Linear(15, 25),
    nn.LeakyReLU(0.01, True)
)
layer2 = nn.Sequential(
    nn.Linear(25, 10),
    nn.LeakyReLU(0.01, True)
)
output_layer = nn.Sequential(
    nn.Linear(10, 1),
    nn.LeakyReLU(0.01, True)
)

# Without Dropout
a1_no_dropout = layer1(X)
print(f"Without Dropout - There are {torch.sum(a1_no_dropout == 0)} dropped units in layer 1.")

# Inverted Dropout from scratch
keep_prop = 0.8  # Removing 20% of all units
dropout_mask = torch.rand(a1_no_dropout.size()) < keep_prop
a1_scratch = layer1(X)
a1_scratch *= dropout_mask
a1_scratch /= keep_prop
num_dropped_scratch = torch.sum(a1_scratch == 0)
print(f"With Dropout - There are {num_dropped_scratch} dropped units in layer 1.")

# Inverted Dropout using nn.Dropout
layer1.add_module("2", nn.Dropout(1-keep_prop, True))
a1_pytorch = layer1(X)
num_dropped_pytorch = torch.sum(a1_pytorch == 0)
print(f"With nn.Dropout - There are {num_dropped_pytorch} dropped units in layer 1.")

assert num_dropped_scratch == num_dropped_pytorch

"""
Note:
    - Those 2 examples (from scratch and pytorch) are not the same
    - They are just for demonstration purposes (i.e., they drop the same number of units in layer 1)
    - In practice, just use nn.Dropout
"""