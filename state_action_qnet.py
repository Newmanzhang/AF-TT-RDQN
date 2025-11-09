
import torch
import torch.nn as nn
from state_encoder import StateEncoder
from noisy_linear import NoisyLinear

class StateActionQNet(nn.Module):
    def __init__(self, state_dim, vocab_size, M, N, token_len, device, n_atoms=51, d_embed=128, use_struct_enc=True):
        super().__init__()
        self.device = device
        self.M = M
        self.n_atoms = n_atoms 
        self.state_enc = StateEncoder(
            state_dim, embed_dim=d_embed, M=M, N=N,
            use_transformer=use_struct_enc
        ).to(device)
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.pos_embedding = nn.Embedding(token_len, d_embed)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=4, dim_feedforward=4 * d_embed,
                                                 batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        USE_NOISY = True
        Linear = NoisyLinear if USE_NOISY else nn.Linear

        self.adv_heads = nn.ModuleList([
            nn.Sequential(
                Linear(d_embed * 2, d_embed),
                nn.ReLU(),
                Linear(d_embed, self.n_atoms) 
            ) for _ in range(M)
        ])

        self.value_head = nn.Sequential(
            Linear(d_embed, d_embed),
            nn.ReLU(),
            Linear(d_embed, self.n_atoms)
        )

    def reset_noise(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "reset_noise"):
                m.reset_noise()

    def forward(self, state_vec, token_batch, ship_token_masks):
        B, L = token_batch.shape
        state_emb = self.state_enc(state_vec)  # (B, D)
        tok_emb = self.token_embedding(token_batch)
        pos_idx = torch.arange(L, device=token_batch.device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embedding(pos_idx)
        x = tok_emb + pos_emb
        x = self.transformer(x)  # (B, L, D)

        adv_list = []
        for i in range(self.M):
            mask_i = ship_token_masks[:, i, :]
            x_i = self.masked_average(x, mask_i, dim=1)
            x_i_cat = torch.cat([x_i, state_emb], dim=-1)
            a_i = self.adv_heads[i](x_i_cat)
            adv_list.append(a_i)


        adv_stack = torch.stack(adv_list, dim=1)

        adv_sum = adv_stack.sum(dim=1)
        adv_mean = adv_stack.mean(dim=1)

        value = self.value_head(state_emb)

        q_total = value.unsqueeze(1) + adv_sum.unsqueeze(1) - adv_mean.unsqueeze(1)
        q_total = q_total.squeeze(1)

        return q_total

    def masked_average(self, x, mask, dim):
        mask = mask.float()
        return (x * mask.unsqueeze(-1)).sum(dim=dim) / (mask.sum(dim=dim, keepdim=True) + 1e-8)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))