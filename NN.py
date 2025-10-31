


import torch
import torch.nn as nn


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
            input = torch.cat([h, log_norm], dim=1)     # concat
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