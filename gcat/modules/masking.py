"""Cell/node masking used during GCAT training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.distributions as td

from .sparse import simple_mask


def drop_nodes(x_dict, drop_node_rate, max_batch_size):
    x = x_dict["x_seq"]
    n_nodes = x.shape[0]
    device = x.device
    if n_nodes == 0:
        return x_dict

    if "seed_mask" in x_dict:
        seed_mask = x_dict["seed_mask"]
        if not torch.is_tensor(seed_mask):
            seed_mask = torch.as_tensor(seed_mask, device=device)
        seed_mask = seed_mask.to(device=device, dtype=torch.bool)
        if seed_mask.numel() != n_nodes:
            seed_mask = torch.zeros(n_nodes, device=device, dtype=torch.bool)
    elif "seed_size" in x_dict:
        seed_size = int(x_dict["seed_size"])
        seed_mask = torch.arange(n_nodes, device=device) < seed_size
    else:
        seed_mask = torch.zeros(n_nodes, device=device, dtype=torch.bool)

    seed_idx = torch.nonzero(seed_mask, as_tuple=False).view(-1)
    neigh_idx = torch.nonzero(~seed_mask, as_tuple=False).view(-1)
    keep_neigh_num = int(neigh_idx.numel() * (1.0 - float(drop_node_rate)))
    if max_batch_size is not None and int(max_batch_size) > 0:
        keep_neigh_num = min(keep_neigh_num, max(int(max_batch_size) - int(seed_idx.numel()), 0))
    if neigh_idx.numel() > 0 and keep_neigh_num > 0:
        keep_neigh = neigh_idx[torch.randperm(neigh_idx.numel(), device=device)[:keep_neigh_num]]
    else:
        keep_neigh = neigh_idx[:0]
    # Keep all center/seed cells first so downstream filter_to_seeds/order logic stays valid.
    cell_idx = torch.cat([seed_idx, keep_neigh], dim=0) if seed_idx.numel() > 0 else keep_neigh
    if cell_idx.numel() == 0:
        cell_idx = torch.arange(n_nodes, device=device)

    if "edge_index" in x_dict:
        edge_index = x_dict["edge_index"].long()
        edge_weight = x_dict.get("edge_weight", None)

        node_keep_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        node_keep_mask[cell_idx] = True
        edge_keep_mask = node_keep_mask[edge_index[0]] & node_keep_mask[edge_index[1]]
        edge_index = edge_index[:, edge_keep_mask]
        if edge_weight is not None:
            edge_weight = edge_weight[edge_keep_mask]

        new_index = -torch.ones(n_nodes, dtype=torch.long, device=device)
        new_index[cell_idx] = torch.arange(len(cell_idx), device=device)
        edge_index = torch.stack([new_index[edge_index[0]], new_index[edge_index[1]]], dim=0)

        x_dict["edge_index"] = edge_index
        if edge_weight is not None:
            x_dict["edge_weight"] = edge_weight

    if "ccc_program_edge_index" in x_dict:
        program_edge_index = x_dict["ccc_program_edge_index"].long()
        program_score = x_dict.get("ccc_program_score", None)
        node_keep_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        node_keep_mask[cell_idx] = True
        edge_keep_mask = (
            node_keep_mask[program_edge_index[0]]
            & node_keep_mask[program_edge_index[1]]
        )
        program_edge_index = program_edge_index[:, edge_keep_mask]
        if program_score is not None:
            program_score = program_score[edge_keep_mask.to(program_score.device)]
        new_index = -torch.ones(n_nodes, dtype=torch.long, device=device)
        new_index[cell_idx] = torch.arange(len(cell_idx), device=device)
        x_dict["ccc_program_edge_index"] = torch.stack(
            [new_index[program_edge_index[0]], new_index[program_edge_index[1]]],
            dim=0,
        )
        if program_score is not None:
            x_dict["ccc_program_score"] = program_score

    idx_np = cell_idx.detach().cpu().numpy()

    def _select_rows(value):
        if torch.is_tensor(value):
            if value.dim() > 0 and value.shape[0] == n_nodes:
                row_idx = cell_idx.to(value.device)
                if value.is_sparse or value.layout in (torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
                    return value.to_dense().index_select(0, row_idx).to_sparse()
                return value.index_select(0, row_idx)
            return value
        if isinstance(value, np.ndarray):
            return value[idx_np] if value.shape[0] == n_nodes else value
        if isinstance(value, list):
            return [value[int(i)] for i in idx_np] if len(value) == n_nodes else value
        return value

    no_row_keys = {
        "edge_index",
        "edge_weight",
        "ccc_program_edge_index",
        "ccc_program_score",
        "gene_mask",
        "input_gene_mask",
        "gene_list",
    }
    # Only these fields are cell-by-cell matrices.  Do not infer that from a
    # square shape: an expression batch can legitimately have n_cells ==
    # n_genes (for ovarian, 496), in which case slicing both axes corrupts the
    # gene dimension.
    cell_square_keys = {
        "ccc_bias",
        "cell_attn_bias",
        "dist_bias",
        "pair_labels",
        "flow_scores",
        "flow_gate_mask",
    }
    for key, value in list(x_dict.items()):
        if key in no_row_keys:
            continue
        if (
            key in cell_square_keys
            and torch.is_tensor(value)
            and value.dim() >= 2
            and value.shape[0] == n_nodes
            and value.shape[1] == n_nodes
        ):
            row_idx = cell_idx.to(value.device)
            if value.is_sparse or value.layout in (torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
                x_dict[key] = value.to_dense().index_select(0, row_idx).index_select(1, row_idx).to_sparse()
            else:
                x_dict[key] = value.index_select(0, row_idx).index_select(1, row_idx)
            continue
        x_dict[key] = _select_rows(value)
    x_dict["seed_size"] = int(seed_idx.numel())
    x_dict["seed_mask"] = torch.arange(int(cell_idx.numel()), device=device) < int(seed_idx.numel())
    x_dict["_drop_node_cell_idx"] = cell_idx
    x_dict["_drop_node_original_n"] = int(n_nodes)
    return x_dict


def _select_dropnode_rows(value, cell_idx, old_n):
    if value is None or cell_idx is None or old_n is None:
        return value
    if not torch.is_tensor(value):
        return value
    row_idx = cell_idx.to(value.device)
    if value.dim() == 1 and value.shape[0] == old_n:
        return value.index_select(0, row_idx)
    if value.dim() >= 2 and value.shape[0] == old_n:
        return value.index_select(0, row_idx)
    return value


def _select_dropnode_square(value, cell_idx, old_n):
    if value is None or cell_idx is None or old_n is None:
        return value
    if not torch.is_tensor(value):
        return value
    row_idx = cell_idx.to(value.device)
    if value.dim() == 2 and value.shape[0] == old_n and value.shape[1] == old_n:
        return value.index_select(0, row_idx).index_select(1, row_idx)
    if value.dim() >= 3 and value.shape[-2] == old_n and value.shape[-1] == old_n:
        return value.index_select(-2, row_idx).index_select(-1, row_idx)
    return value


class MaskBuilder(nn.Module):
    def __init__(
        self,
        mask_node_rate,
        mask_feature_rate,
        drop_node_rate=0,
        max_batch_size=2000,
        edge_mask=None,
        mask_beta=False,
        st_block_gene_mask_rate=0.0,
    ):
        super().__init__()
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate
        self._edge_mask = edge_mask
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size
        if self._mask_node_rate > 0 and self._mask_feature_rate and mask_beta:
            alpha = 5
            beta = 4 / self._mask_feature_rate + 2 - alpha
            self.beta_dist = td.Beta(alpha, beta)
        self.mask_beta = mask_beta
        self.st_block_gene_mask_rate = float(st_block_gene_mask_rate)
        if not 0.0 <= self.st_block_gene_mask_rate <= 1.0:
            raise ValueError("st_block_gene_mask_rate must be in [0, 1].")

    def update_mask_ratio(self, mask_node_rate, mask_feature_rate):
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate

    def apply_mask(self, x_dict):
        if self.training and self._drop_node_rate > 0:
            drop_nodes(x_dict, self._drop_node_rate, self._max_batch_size)
        use_random_mask = self.training and self._mask_node_rate > 0
        use_st_block_mask = self.training and self.st_block_gene_mask_rate > 0
        if use_random_mask or use_st_block_mask:
            if "x_masked_seq" in x_dict:
                x = x_dict["x_masked_seq"]
            else:
                x = x_dict["x_seq"]
            mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
            if use_random_mask:
                if self.mask_beta:
                    mask_ratio = self.beta_dist.sample((x.shape[0],)).to(x.device)
                    mask_ratio[mask_ratio > 0.9] = 0.9
                    num_nodes = x.shape[0]
                    perm = np.random.permutation(num_nodes)
                    num_mask_nodes = int(self._mask_node_rate * num_nodes)
                    keep_nodes = perm[num_mask_nodes:]
                    random_mask = torch.rand(*x.shape, device=x.device) <= mask_ratio.unsqueeze(-1)
                    random_mask[keep_nodes] = False
                else:
                    num_nodes = x.shape[0]
                    perm = np.random.permutation(num_nodes)
                    num_mask_nodes = int(self._mask_node_rate * num_nodes)
                    keep_nodes = perm[num_mask_nodes:]
                    random_mask = torch.rand(*x.shape, device=x.device) <= self._mask_feature_rate
                    random_mask[keep_nodes] = False
                mask |= random_mask

            block_mask = torch.zeros_like(mask)
            if use_st_block_mask and "source" in x_dict:
                source = torch.as_tensor(x_dict["source"], device=x.device, dtype=torch.bool).reshape(-1)
                if source.numel() != x.shape[0]:
                    raise ValueError(
                        f"source length {source.numel()} does not match batch size {x.shape[0]}."
                    )
                st_rows = ~source
                if bool(st_rows.any().item()):
                    gene_mask = x_dict.get("gene_mask", None)
                    if gene_mask is None:
                        observed = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
                    else:
                        observed = torch.as_tensor(gene_mask, device=x.device, dtype=torch.bool)
                        if observed.dim() == 2:
                            observed = observed[st_rows].all(dim=0)
                        elif observed.dim() != 1:
                            raise ValueError(
                                f"gene_mask must be 1D or 2D, got {tuple(observed.shape)}."
                            )
                        if observed.numel() != x.shape[1]:
                            raise ValueError(
                                f"gene_mask length {observed.numel()} does not match gene dim {x.shape[1]}."
                            )
                    candidates = torch.nonzero(observed, as_tuple=False).flatten()
                    if candidates.numel() > 0:
                        block_count = max(
                            1,
                            int(round(self.st_block_gene_mask_rate * int(candidates.numel()))),
                        )
                        block_count = min(block_count, int(candidates.numel()))
                        selected = candidates[
                            torch.randperm(candidates.numel(), device=x.device)[:block_count]
                        ]
                        selected_mask = torch.zeros_like(observed)
                        selected_mask[selected] = True
                        block_mask = st_rows.unsqueeze(1) & selected_mask.unsqueeze(0)
                        mask |= block_mask
            x_dict["st_block_gene_mask"] = block_mask
            if not x.is_sparse:
                x = x.to_sparse()
            x = x.coalesce()
            masked_x_seq = simple_mask(x, mask)
            x_dict["masked_x_seq"] = masked_x_seq
            if "x_bin" in x_dict:
                x_bin = x_dict["x_bin"]
                if not x_bin.is_sparse:
                    x_bin = x_bin.to_sparse()
                x_bin = x_bin.coalesce()
                x_dict["masked_x_bin"] = simple_mask(x_bin, mask)
            x_dict["input_mask"] = mask.int()
        else:
            x_dict["input_mask"] = torch.ones(*x_dict["x_seq"].shape, device=x_dict["x_seq"].device).int()
        return x_dict
