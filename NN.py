


import torch
import torch.nn as nn
import numpy as np
import random

from own_plotting import plots_vort
import matplotlib.pyplot as plt
plot_debug= False

def save_checkpoint(path, gen_sde, optim, iteration):
    checkpoint = {
        "iteration": iteration,
        "model": gen_sde.state_dict(),
        "optimizer": optim.state_dict(),
        "torch_rng": torch.get_rng_state().cpu(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, gen_sde, optim, device):
    # Add weights_only=False to allow optimizer / RNG / numpy objects
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    gen_sde.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optimizer'])

    # --- restore RNG states correctly ---
    torch_rng = checkpoint["torch_rng"]
    if not torch_rng.dtype == torch.uint8:
        torch_rng = torch_rng.to(torch.uint8)
    torch.set_rng_state(torch_rng.cpu())  # MUST BE CPU

    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])

    start_iter = checkpoint['iteration']
    print(f"Resuming from iteration {start_iter+1}")
    return start_iter


#  Define models

### 2.2. Define MLP
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)*x
    

class NormalizeLogRadius(nn.Module):
    """Non-learnable preprocessing layer:
       x ↦ (x/||x||, log||x||).
    """
    def __init__(self, eps = 1e-6): 
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch, d)
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm + self.eps
        x_normalized = x / norm
        log_norm = torch.log(norm)
        return x_normalized, log_norm
    

class MLP(nn.Module):
    def __init__(self,
                input_dim=2,
                index_dim=1,
                hidden_dim=128,
                act=Swish(),
                premodule = None,
                ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act

        self.output_dim = self.input_dim # represent the a vector field so input dim = output dim

        assert premodule is None or premodule in ["NormalizeLogRadius"]
        self.premodule = premodule
        if premodule == "NormalizeLogRadius": 
            self.pre = NormalizeLogRadius()                                    # non - learnable
            self.learnable_network_input_dim = self.input_dim +  1
        else: # premodule is None
            self.pre = None
            self.learnable_network_input_dim = self.input_dim

        self.main = nn.Sequential(
            nn.Linear(self.learnable_network_input_dim+index_dim, hidden_dim),  # input layer
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, self.output_dim),                             # output layer
        )

    def forward(self, input, t):
        # init
        sz = input.size()
        input = input.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()

        # forward
        if self.pre is not None:
            h, log_norm = self.pre(input)                   # (batch, learnable_network_input_dim)
            input = torch.cat([h, log_norm], dim=-1)     # concat
        h = torch.cat([input, t], dim=1)     # concat
        output = self.main(h)                    # forward
        return output.view(*sz)

### 2.3. Define evaluate function (compute ELBO)
@torch.no_grad()
def evaluate(gen_sde, x_test):
    gen_sde.eval()
    num_samples_ = x_test.size(0)
    test_elbo = gen_sde.elbo_random_t_slice(x_test)
    gen_sde.train()
    return test_elbo.mean(), test_elbo.std() / num_samples_ ** 0.5