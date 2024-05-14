import torch
import nni
from torch import nn
from nni.nas.nn.pytorch import ModelSpace, Repeat, MutableLinear, LayerChoice


class MLP(ModelSpace, label_prefix = 'MLP'):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        n_blocks = nni.choice('n_blocks', list(range(1, 11)))
        d_block = nni.choice('d_block', [2 ** (i + 5) for i in range(5)])

        self.in_block = nn.Sequential(
            MutableLinear(d_in, d_block),
            LayerChoice([nn.ReLU(), nn.GELU(), nn.SiLU()], label='in_act'),
            nn.Dropout(dropout)
        )

        block = nn.Sequential(
            MutableLinear(d_block, d_block),
            LayerChoice([nn.ReLU(), nn.GELU(), nn.SiLU()], label='blocks_act'),
            nn.Dropout(dropout)
        )

        self.blocks = Repeat(block, n_blocks)
        self.out_block = MutableLinear(d_block, d_out)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.in_block(inp)
        x = self.blocks(x)
        out = self.out_block(x)
        return out
