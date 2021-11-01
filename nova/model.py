'''
    The model you create.
'''

# ### If you need to import from other folder ###
import os.path as path
import sys

dirname__ = path.dirname(path.abspath(__file__))
sys.path.append(path.join(dirname__, '..'))
# ### ###

import torch.nn as nn
import torch.nn.functional as F


class NETWORK_NAME(nn.Module):
    def __init__(self) -> None:
        super(NETWORK_NAME, self).__init__()

        # Build you network here.

    def forward(self, x):
        # Forward your gradients here.
        return x


if __name__ == "__main__":
    print(NETWORK_NAME())
