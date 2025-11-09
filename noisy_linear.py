import math, torch, torch.nn as nn

class NoisyLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_f, self.out_f = in_features, out_features


        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.empty(out_features))


        self.weight_sig = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sig   = nn.Parameter(torch.empty(out_features))


        self.register_buffer("eps_w", torch.zeros(out_features, in_features))
        self.register_buffer("eps_b", torch.zeros(out_features))

        self.reset_parameters(sigma0)
        self.reset_noise()

    def reset_parameters(self, sigma0):
        bound = 1 / math.sqrt(self.in_f)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu,   -bound, bound)
        nn.init.constant_(self.weight_sig, sigma0 / math.sqrt(self.in_f))
        nn.init.constant_(self.bias_sig,  sigma0 / math.sqrt(self.out_f))

    def _f(self, x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in  = torch.randn(self.in_f,  device=self.weight_mu.device)
        eps_out = torch.randn(self.out_f, device=self.weight_mu.device)
        self.eps_w = self._f(eps_out).ger(self._f(eps_in))
        self.eps_b = self._f(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sig * self.eps_w
            b = self.bias_mu   + self.bias_sig  * self.eps_b
        else:
            w, b = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, w, b)
