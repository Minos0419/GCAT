"""Expression embedding used by GCAT."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GeneExpressionEmbedding(nn.Module):
    def __init__(self, gene_list, hidden_dim: int):
        super().__init__()
        self.gene_list = list(gene_list)
        self.gene_index = {gene: index for index, gene in enumerate(self.gene_list)}
        self.embedding = nn.Parameter(
            torch.randn(len(self.gene_list), hidden_dim, dtype=torch.float32) * 0.005
        )

    def forward(self, x_dict, input_gene_list=None):
        x = x_dict.get("masked_x_seq", x_dict["x_seq"])
        if "dropout" in x_dict:
            x = x.coalesce()
            values = torch.distributions.binomial.Binomial(
                x.values().float(), x_dict["dropout"]
            ).sample()
            x = torch.sparse_coo_tensor(x.indices(), values, x.shape, device=x.device)

        x = torch.log1p(x)
        if input_gene_list is None:
            if x.shape[1] != len(self.gene_list):
                raise ValueError("input_gene_list is required when the input gene set differs from config.json")
            gene_index = torch.arange(x.shape[1], device=x.device)
        else:
            gene_index = torch.tensor(
                [self.gene_index[gene] for gene in input_gene_list if gene in self.gene_index],
                device=x.device,
                dtype=torch.long,
            )
            x_dict["input_gene_mask"] = gene_index

        gene_table = F.embedding(gene_index, self.embedding)
        x_dense = x.to_dense() if x.is_sparse else x
        cell_embedding = (x_dense.float() @ gene_table.float()).to(self.embedding.dtype)
        gene_embedding = (
            x_dense.to(self.embedding.dtype).unsqueeze(-1)
            * self.embedding[gene_index].unsqueeze(0)
        )
        return cell_embedding, gene_embedding


class GCATEmbedding(nn.Module):
    def __init__(self, gene_list, hidden_dim: int, dropout: float):
        super().__init__()
        self.cell_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        # The historical constructor sampled one 512-D positional fallback here.
        # It never participated in GCAT's forward pass, but preserving this single
        # draw keeps the retained parameters identical for the same training seed.
        torch.randn(hidden_dim)
        self.gene_embedding = GeneExpressionEmbedding(gene_list, hidden_dim)

    def forward(self, x_dict, input_gene_list=None):
        cell_embedding, gene_embedding = self.gene_embedding(x_dict, input_gene_list)
        return gene_embedding, self.cell_projection(cell_embedding)
