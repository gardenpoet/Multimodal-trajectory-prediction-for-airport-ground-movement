import torch    
import torch.nn as nn
from easydict import EasyDict
from torch.nn import functional as F
from typing import Tuple

class GMM(nn.Module):
    """ Gaussian Mixture Model module with velocity + acceleration outputs. """
    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        self.config = config
    
        # Derived values
        self.num_futures = config.num_futures
        gmm_embd = config.in_size // config.num_futures
        self.out_dim = int((config.num_dims - 1) // 2)   # D/2 if dims = [mu(acc), sigma(acc), score]

        # Shared feature extractor
        self.future_heads = nn.Sequential(
            nn.Linear(gmm_embd, 4 * gmm_embd),
            nn.GELU(),
            nn.Linear(4 * gmm_embd, gmm_embd, bias=False)
        )

        # Acceleration head
        self.acc_head = nn.Linear(gmm_embd, config.num_dims, bias=False)

        # Velocity head (new)
        self.vel_head = nn.Linear(gmm_embd, self.out_dim, bias=False)
    

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Model's forward function with debug prints for tensor shapes.
        """
    
        B, A, T, C = x.size()
    
        # Check if C divides evenly by num_futures
        if C % self.num_futures != 0:
            print(f"[WARNING] C ({C}) is not divisible by num_futures ({self.num_futures})!")
    
        # Reshape input for decoding
        x = x.view(B, A, T, self.num_futures, C // self.num_futures)
    
        # Pass through future heads
        out = self.future_heads(x)
    
        # Extract predicted components
        pred_scores = F.softmax(out[..., -1].mean(-2), dim=-1)
    
        mu = out[..., :self.out_dim]

        raw_sigma = out[..., self.out_dim:(self.out_dim*2)]
        sigma = F.softplus(raw_sigma) + 1e-3
        # sigma = torch.clamp(sigma, min=0.00005, max=0.001)
        # sigma = torch.exp(out[..., self.out_dim:(self.out_dim * 2)])
    
        return pred_scores, mu, sigma
