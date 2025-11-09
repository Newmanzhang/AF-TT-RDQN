import torch, math
import torch.nn as nn
from typing import Tuple

class _MLPRes(nn.Module):

    def __init__(self, state_dim: int, embed_dim: int, n_blocks: int = 3):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_dim, 2048), nn.LayerNorm(2048), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.ReLU()
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.LayerNorm(1024)
            ))
        self.res_blocks = nn.ModuleList(blocks)
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for blk in self.res_blocks:
            x = torch.relu(blk(x) + x)
        return self.final(x)


class _StateTransformer(nn.Module):

    SHIP_F = 9
    EDGE_F = 6
    def __init__(self, state_dim: int, embed_dim: int,
                 M: int, N: int,
                 n_heads: int = 4, n_layers: int = 2, ff_mult: int = 4):
        super().__init__()
        self.M, self.N = M, N
        token_dim = embed_dim

        self.ship_proj  = nn.Linear(self.SHIP_F, token_dim)
        self.edge_proj  = nn.Linear(self.EDGE_F, token_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=ff_mult * token_dim,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)


        rest_dim = state_dim - (self.SHIP_F * M + self.EDGE_F * N)
        if rest_dim <= 0:
            raise ValueError("state_dim small")
        self.global_mlp = nn.Sequential(
            nn.Linear(rest_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, token_dim)
        )

        self.out = nn.Sequential(
            nn.Linear(token_dim, embed_dim)
        )


    def _split(self, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B = x.size(0)
        s_end = self.M * self.SHIP_F
        e_end = s_end + self.N * self.EDGE_F
        ships = x[:, :s_end].view(B, self.M, self.SHIP_F)
        edges = x[:, s_end:e_end].view(B, self.N, self.EDGE_F)
        rest  = x[:, e_end:]
        return ships, edges, rest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ships, edges, rest = self._split(x)

        ship_tok = self.ship_proj(ships)
        edge_tok = self.edge_proj(edges)
        seq = torch.cat([ship_tok, edge_tok], dim=1)

        seq = self.transformer(seq)
        rel_emb = seq.mean(dim=1)

        global_emb = self.global_mlp(rest)
        return self.out(rel_emb + global_emb)


class StateEncoder(nn.Module):

    def __init__(self,
                 state_dim: int,
                 embed_dim: int = 128,
                 M: int = 5,
                 N: int = 3,
                 use_transformer: bool = True):
        super().__init__()
        if use_transformer:
            self.enc = _StateTransformer(state_dim, embed_dim, M, N)
        else:
            self.enc = _MLPRes(state_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)
