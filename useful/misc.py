import torch.nn as nn

# Count parameters of network.
def get_param_count(net: nn.Module):
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    return {'trainable': trainable, 'total': total}
