from typing import Dict
import torch
from torch import nn
import itertools


class TransformerEncoderExpert(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        return self.transformer_encoder(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, hidden_dim=256):
        super().__init__()
        self.top_k = top_k
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x):
        logits = self.network(x)
        if self.training and self.top_k < logits.size(1):
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
            top_k_gates = torch.softmax(top_k_logits, dim=-1)
            # zeros = torch.zeros_like(logits)
            zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
        else:
            gates = torch.softmax(logits, dim=-1)
        return gates


class Expert_General(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.expert_fusion = TransformerEncoderExpert(d_model, nhead, num_layers)

    def forward(self, inputs):
        return self.expert_fusion(inputs[0])


class ECSModule(nn.Module):
    def __init__(
        self,
        d_model,
        num_experts_per_combination,
        num_experts,
        nhead,
        num_layers,
        num_embs,
        top_k=2,
    ):
        super().__init__()
        self.num_embs = num_embs
        self.d_model = d_model
        self.top_k = top_k
        self.expert_combinations = []
        self.experts = nn.ModuleList()

        combinations = list(
            itertools.chain(
                [(i,) for i in range(num_embs)],  # Single embeddings
                [tuple(range(num_embs))],  # Full combination
            )
        )

        expert_idx = 0
        for combination in combinations:
            for _ in range(num_experts_per_combination):
                if expert_idx >= num_experts:
                    break
                self.expert_combinations.append(combination)
                self.experts.append(Expert_General(d_model, nhead, num_layers))
                expert_idx += 1

        self.gating_network = GatingNetwork(
            d_model * num_embs, len(self.experts), top_k=top_k
        )

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb_list = [embeddings[key] for key in sorted(embeddings.keys())]

        comb_inputs = []
        for combination in self.expert_combinations:
            combined = torch.cat([emb_list[i] for i in combination], dim=1)
            comb_inputs.append(combined)

        expert_outputs = []
        for expert, input_tensor in zip(self.experts, comb_inputs):
            output = expert([input_tensor])  # [batch, seq, d_model]
            expert_outputs.append(output[:, 0, :])  # CLS token

        expert_outputs = torch.stack(
            expert_outputs, dim=1
        )  # [batch, num_experts, d_model]

        cls_embeddings = torch.cat([emb[:, 0, :] for emb in emb_list], dim=-1)
        gating_weights = self.gating_network(cls_embeddings)

        final_output = torch.bmm(
            gating_weights.unsqueeze(1),  # [batch, 1, num_experts]
            expert_outputs,  # [batch, num_experts, d_model]
        ).squeeze(1)

        return final_output
