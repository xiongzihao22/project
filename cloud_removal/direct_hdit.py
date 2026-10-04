"""Parameter-matched, deterministic HDiT regression without a teacher."""
import torch
from torch import nn
from .hdit_factory import build_hdit


class DirectHDiT(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.backbone = build_hdit(spec)

    def forward(self, cloudy):
        # Keep the existing six-channel backbone; the other stream is always zero.
        # Constant positive time avoids log(0), without adding diffusion/noise.
        return self.backbone(torch.cat((torch.zeros_like(cloudy), cloudy), dim=1),
                             torch.ones(cloudy.shape[0], device=cloudy.device))
