from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import random
import torch
import torch_sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

try:
    import anndata as ad
except Exception:  # pragma: no cover - optional at import time for tooling
    ad = Any  # type: ignore[misc,assignment]


# SPATIAL_PLATFORM_LIST = {"cosmx", "merfish", "seqFISH", "STARmap", "ISS", "xenium", "visium"}

SPATIAL_PLATFORM_LIST = {"cosmx", "merfish", "seqfish", "starmap", "iss", "xenium", "visium"}

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_csr(x: Union[np.ndarray, sp.spmatrix]) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def _log1p_csr(x: sp.csr_matrix) -> sp.csr_matrix:
    x = x.copy()
    x.data = np.log1p(x.data)
    return x


def _coords_from_adata(
    adata: "ad.AnnData",
    coords_key: str = "spatial",
    obs_cols: Tuple[str, str] = ("x_FOV_px", "y_FOV_px"),
) -> Optional[np.ndarray]:
    if hasattr(adata, "obsm") and coords_key in adata.obsm:
        coords = np.asarray(adata.obsm[coords_key])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2].astype(np.float32, copy=False)
    if hasattr(adata, "obs") and all(c in adata.obs for c in obs_cols):
        coords = np.asarray(adata.obs[list(obs_cols)].to_numpy())
        if coords.ndim == 2 and coords.shape[1] == 2:
            return coords.astype(np.float32, copy=False)
    return None


def _build_knn_edges(
    points: np.ndarray,
    k: int,
    *,
    metric: str = "euclidean",
    symmetrize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 2:
        raise ValueError(f"`points` must be 2D, got shape {points.shape}.")
    n = points.shape[0]
    if n == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
        return edge_index, edge_weight

    k_eff = int(min(max(k, 0), max(n - 1, 0)))
    if k_eff == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
        return edge_index, edge_weight

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(points)
    neigh = nn.kneighbors(points, return_distance=False)

    src_list: List[np.ndarray] = []
    dst_list: List[np.ndarray] = []
    for i in range(n):
        nbrs = neigh[i]
        nbrs = nbrs[nbrs != i]  # drop self
        if nbrs.size == 0:
            continue
        src_list.append(np.full(nbrs.shape, i, dtype=np.int64))
        dst_list.append(nbrs.astype(np.int64, copy=False))

    if not src_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
        return edge_index, edge_weight

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    if symmetrize:
        src_orig = src
        dst_orig = dst
        src = np.concatenate([src_orig, dst_orig])
        dst = np.concatenate([dst_orig, src_orig])

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


def _coo_from_adj(adj: Union[np.ndarray, sp.spmatrix]) -> sp.coo_matrix:
    if sp.issparse(adj):
        return adj.tocoo()
    return sp.coo_matrix(adj)


def build_spatial_graph(
    adata: "ad.AnnData",
    *,
    k: int = 4,
    coords_key: str = "spatial",
    obs_cols: Tuple[str, str] = ("x_FOV_px", "y_FOV_px"),
    adj_key: Optional[str] = None,
    symmetrize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if adj_key is not None and hasattr(adata, "obsp") and adj_key in adata.obsp:
        adj = _coo_from_adj(adata.obsp[adj_key])
        edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
        edge_weight = torch.tensor(adj.data, dtype=torch.float32)
        coords = _coords_from_adata(adata, coords_key=coords_key, obs_cols=obs_cols)
        if coords is None:
            coords = np.full((adata.n_obs, 2), -1.0, dtype=np.float32)
        return edge_index, edge_weight, coords

    coords = _coords_from_adata(adata, coords_key=coords_key, obs_cols=obs_cols)
    if coords is None:
        raise ValueError(
            "Spatial graph requested, but no spatial coordinates found. "
            f"Tried `adata.obsm['{coords_key}']` and `adata.obs[{list(obs_cols)}]`."
        )
    edge_index, edge_weight = _build_knn_edges(coords, k=k, symmetrize=symmetrize)
    return edge_index, edge_weight, coords


def build_expression_pca_knn_graph(
    adata: "ad.AnnData",
    *,
    k: int = 10,
    n_components: int = 50,
    log1p: bool = True,
    symmetrize: bool = True,
    random_state: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = _to_csr(adata.X)
    if log1p:
        x = _log1p_csr(x)

    n = x.shape[0]
    if n <= 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
        return edge_index, edge_weight

    n_components_eff = int(min(max(n_components, 2), min(n - 1, x.shape[1] - 1)))
    if sp.issparse(x):
        reducer = TruncatedSVD(n_components=n_components_eff, random_state=random_state)
        emb = reducer.fit_transform(x)
    else:
        reducer = PCA(n_components=n_components_eff, random_state=random_state)
        emb = reducer.fit_transform(np.asarray(x))

    edge_index, edge_weight = _build_knn_edges(emb, k=k, symmetrize=symmetrize)
    return edge_index, edge_weight


def scipy_csr_to_torch_sparse(x: sp.csr_matrix) -> torch.Tensor:
    x = x.tocsr()
    indptr = torch.tensor(x.indptr, dtype=torch.int64)
    indices = torch.tensor(x.indices, dtype=torch.int64)
    data = torch.tensor(x.data, dtype=torch.float32)
    return torch.sparse_csr_tensor(indptr, indices, data, size=x.shape).to_sparse().coalesce()


def aggregate_one_hop_neighbors(
    x: sp.csr_matrix,
    edge_index: torch.Tensor,
    *,
    agg: Literal["sum", "mean"] = "mean",
) -> sp.csr_matrix:
    n = x.shape[0]
    if edge_index.numel() == 0:
        return sp.csr_matrix(x.shape, dtype=x.dtype)

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    data = np.ones_like(src, dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    x_neighbors = adj.T.dot(x)
    if agg == "mean":
        deg_in = np.asarray(adj.sum(axis=0)).ravel().astype(np.float32)
        deg_in[deg_in == 0] = 1.0
        x_neighbors = x_neighbors.multiply(1.0 / deg_in.reshape(-1, 1))
    return x_neighbors.tocsr()


def _make_gene_mask_from_gene_list(
    gene_list: Sequence[str],
    observed_genes: Optional[Sequence[str]],
) -> torch.Tensor:
    if observed_genes is None:
        return torch.ones(len(gene_list), dtype=torch.bool)
    observed = set(observed_genes)
    mask = torch.tensor([g in observed for g in gene_list], dtype=torch.bool)
    return mask


@dataclass(frozen=True)
class OneHopLoaderConfig:
    adata: "ad.AnnData"
    split_field: str = "split"
    train_split: Union[str, Sequence[str], None] = "train"
    val_split: Union[str, Sequence[str], None] = "valid"
    test_split: Union[str, Sequence[str], None] = "test"
    batch_size: int = 128
    num_neighbors: int = 10
    num_workers: int = 0
    shuffle_train: bool = True
    coords_key: str = "spatial"
    coords_obs_cols: Tuple[str, str] = ("x_FOV_px", "y_FOV_px")
    spatial_k: int = 4
    sc_knn_k: int = 10
    sc_pca_n_components: int = 50
    sc_log1p: bool = True
    spatial_adj_key: Optional[str] = None
    neighbor_agg: Optional[Literal["sum", "mean"]] = "mean"
    max_nodes_per_batch: Optional[int] = None
    order_required: bool = False
    covariate_fields: Tuple[str, ...] = ()
    covariate_categories: Dict[str, Sequence[str]] = field(default_factory=dict)
    label_fields: Tuple[str, ...] = ()
    observed_genes: Optional[Sequence[str]] = None
    gp_targets_mask_key: str = "nichecompass_gp_targets"
    gp_sources_mask_key: str = "nichecompass_gp_sources"
    gp_denovo_dim: int = 0


class NCOneHopNeighborDataloader:
    """
    One-hop neighbor dataloader that yields dict batches compatible with the
    original TranscriptomicDataset key naming (x_seq, coord, edge_index, ...).
    """

    def __init__(
        self,
        *,
        adata: "ad.AnnData",
        global_node_ids: Optional[np.ndarray],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        input_nodes: Optional[np.ndarray],
        gene_mask: torch.Tensor,
        gene_list: List[str],
        split_field: str,
        covariate_fields: Sequence[str],
        covariate_categories: Optional[Mapping[str, Sequence[str]]] = None,
        label_fields: Sequence[str],
        neighbor_agg: Optional[Literal["sum", "mean"]],
        use_gp: bool,
        gp_targets_mask_key: str,
        gp_sources_mask_key: str,
        gp_denovo_dim: int,
        order_required: bool,
        batch_size: int,
        num_neighbors: int,
        shuffle: bool,
        num_workers: int,
        coords: Optional[np.ndarray],
        coords_key_present: bool,
        max_nodes_per_batch: Optional[int] = None,
    ) -> None:
        self.adata = adata
        self._global_node_ids = global_node_ids
        self._x_csr = _to_csr(adata.X)
        self.gene_list = gene_list
        self.gene_mask = gene_mask
        self.split_field = split_field
        self.covariate_fields = tuple(covariate_fields)
        self.covariate_categories = {
            str(name): [str(value) for value in values]
            for name, values in (covariate_categories or {}).items()
        }
        self.label_fields = tuple(label_fields)
        self.neighbor_agg = neighbor_agg
        self.use_gp = use_gp
        self.gp_denovo_dim = int(max(gp_denovo_dim, 0))
        self.max_nodes_per_batch = int(max_nodes_per_batch) if max_nodes_per_batch else None
        self.order_required = bool(order_required)

        self._coords = coords
        self._coords_key_present = coords_key_present

        self._covariates_encoded: Dict[str, torch.Tensor] = {}
        for c in self.covariate_fields:
            if c in adata.obs:
                values = adata.obs[c].astype(str).to_numpy()
                categories = self.covariate_categories.get(c)
                if categories:
                    mapping = {value: index for index, value in enumerate(categories)}
                    encoded = np.asarray([mapping[value] for value in values], dtype=np.int64)
                else:
                    encoded = pd_factorize(values)
                self._covariates_encoded[c] = torch.tensor(encoded, dtype=torch.long)

        self._labels_encoded: Dict[str, torch.Tensor] = {}
        for l in self.label_fields:
            if l in adata.obs:
                self._labels_encoded[l] = torch.tensor(
                    pd_factorize(adata.obs[l].to_numpy()), dtype=torch.long
                )

        self._gp_targets_mask: Optional[np.ndarray] = None
        self._gp_sources_mask: Optional[np.ndarray] = None
        if use_gp and hasattr(adata, "varm"):
            if gp_targets_mask_key in adata.varm:
                self._gp_targets_mask = np.asarray(adata.varm[gp_targets_mask_key], dtype=np.float32)
            if gp_sources_mask_key in adata.varm:
                self._gp_sources_mask = np.asarray(adata.varm[gp_sources_mask_key], dtype=np.float32)

        pyg_data = Data(
            x=torch.arange(adata.n_obs, dtype=torch.long).unsqueeze(1),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        if input_nodes is not None:
            input_nodes_t = torch.from_numpy(input_nodes.astype(np.int64, copy=False))
        else:
            input_nodes_t = None

        self._loader = NeighborLoader(
            pyg_data,
            num_neighbors=[int(num_neighbors)],
            batch_size=int(batch_size),
            input_nodes=input_nodes_t,
            shuffle=bool(shuffle),
            directed=False,
            num_workers=int(num_workers),
        )

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for batch in self._loader:
            yield self._batch_to_dict(batch)

    def __len__(self) -> int:
        try:
            return len(self._loader)
        except TypeError:
            return 0

    def _batch_to_dict(self, batch: Data) -> Dict[str, Any]:
        # `n_id` maps local batch node indices -> global node indices.
        n_id_t = batch.n_id.detach().cpu()
        edge_index = batch.edge_index.long()
        edge_weight = getattr(batch, "edge_weight", torch.ones(edge_index.size(1))).float()
        seed_size = int(getattr(batch, "batch_size", 0))

        if self.max_nodes_per_batch is not None:
            n_id_t, edge_index, edge_weight, seed_size = _cap_batch_nodes(
                n_id=n_id_t,
                edge_index=edge_index,
                edge_weight=edge_weight,
                seed_size=seed_size,
                max_nodes=self.max_nodes_per_batch,
            )

        n_id_local = n_id_t.numpy().astype(np.int64, copy=False)
        x_csr = self._x_csr[n_id_local]
        if self._global_node_ids is not None:
            n_id_out = self._global_node_ids[n_id_local]
        else:
            n_id_out = n_id_local

        out: Dict[str, Any] = {
            "x_seq": scipy_csr_to_torch_sparse(x_csr),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "gene_mask": self.gene_mask,
            "gene_list": self.gene_list,
        }
        # Expose which rows are "seed" nodes (center nodes) vs sampled neighbors.
        out["seed_size"] = seed_size
        out["seed_mask"] = torch.arange(int(n_id_t.numel()), dtype=torch.long) < seed_size
        out["n_id"] = torch.from_numpy(n_id_out)
        if self.order_required:
            # Order encoding for inference: only seed nodes are expected to be
            # consumed exactly once; neighbors may repeat across batches.
            out["order_list"] = out["n_id"][:seed_size]

        # coords (ST: real coords; sc: -1)
        if self._coords is not None and self._coords_key_present:
            out["coord"] = torch.from_numpy(self._coords[n_id_local]).float()
        else:
            out["coord"] = torch.full((int(n_id_t.numel()), 2), -1.0, dtype=torch.float32)

        # split labels (keep as numpy array of strings for compatibility)
        if self.split_field and self.split_field in self.adata.obs:
            out["split"] = self.adata.obs[self.split_field].to_numpy()[n_id_local]
        else:
            out["split"] = None

        # covariates / labels (encoded ints like the old dataset)
        for c, encoded in self._covariates_encoded.items():
            out[c] = encoded[n_id_local]
        for l, encoded in self._labels_encoded.items():
            out[l] = encoded[n_id_local]

        # extra covs: source / modality / specie / assay (keep as Python lists)
        for cov in ("source", "specie", "assay"):
            if cov in self.adata.obs:
                out[cov] = self.adata.obs[cov].to_numpy()[n_id_local].tolist()

        # NicheCompass-style neighbor aggregation (optional)
        if self.neighbor_agg is not None:
            x_neighbors = aggregate_one_hop_neighbors(
                x_csr, out["edge_index"], agg=self.neighbor_agg
            )
            out["x_seq_agg_neighbors"] = scipy_csr_to_torch_sparse(x_neighbors)

        # Optional GP features: simple gene-set scores based on provided masks.
        if self.use_gp and self._gp_targets_mask is not None:
            # adata.varm masks are shaped (n_genes, n_gps)
            gp_prior = x_csr.dot(self._gp_targets_mask)
            out["gp_prior_features"] = torch.tensor(np.asarray(gp_prior), dtype=torch.float32)
            if self.gp_denovo_dim > 0:
                out["gp_denovo_features"] = torch.zeros(
                    (x_csr.shape[0], self.gp_denovo_dim), dtype=torch.float32
                )

        return out


def pd_factorize(values: np.ndarray) -> np.ndarray:
    """
    Lightweight, dependency-free label encoding compatible with pandas/objects.
    Returns int64 codes starting at 0.
    """
    # Keep NaNs as their own category for stability.
    uniques, codes = np.unique(values.astype("U"), return_inverse=True)
    _ = uniques  # unused, but kept for readability
    return codes.astype(np.int64, copy=False)


def _cap_batch_nodes(
    *,
    n_id: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    seed_size: int,
    max_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Cap the total number of nodes in a NeighborLoader batch while keeping
    seed nodes first. Prefer neighbors that connect to many seeds.
    """
    n_total = int(n_id.numel())
    if max_nodes <= 0 or n_total <= max_nodes:
        return n_id, edge_index, edge_weight, seed_size

    seed_size = int(max(seed_size, 0))
    if seed_size == 0:
        # No concept of seeds: just truncate deterministically.
        keep = torch.arange(max_nodes, dtype=torch.long)
        seed_size_new = 0
    elif seed_size >= max_nodes:
        keep = torch.arange(max_nodes, dtype=torch.long)
        seed_size_new = int(max_nodes)
    else:
        keep_seed = torch.arange(seed_size, dtype=torch.long)
        n_neighbors_keep = int(max_nodes - seed_size)
        cand = torch.arange(seed_size, n_total, dtype=torch.long)
        if cand.numel() == 0 or n_neighbors_keep <= 0:
            keep = keep_seed
        else:
            src = edge_index[0]
            dst = edge_index[1]

            # Count how often each candidate neighbor connects to any seed node.
            # We look at edges where exactly one endpoint is a seed.
            scores = torch.zeros(cand.numel(), dtype=torch.long)
            mask = (src < seed_size) & (dst >= seed_size)
            if mask.any():
                idx = dst[mask] - seed_size
                scores.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
            mask = (dst < seed_size) & (src >= seed_size)
            if mask.any():
                idx = src[mask] - seed_size
                scores.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

            topk = min(n_neighbors_keep, scores.numel())
            if topk <= 0:
                keep = keep_seed
            else:
                # `topk` on scores; tie-breaker by lower index (stable enough).
                _, order = torch.topk(scores, k=topk, largest=True)
                keep_neighbors = cand[order]
                keep = torch.cat([keep_seed, keep_neighbors], dim=0)
                keep, _ = keep.sort()
        seed_size_new = seed_size

    # Remap edges to the kept nodes.
    mapping = torch.full((n_total,), -1, dtype=torch.long)
    mapping[keep] = torch.arange(keep.numel(), dtype=torch.long)
    src_new = mapping[edge_index[0]]
    dst_new = mapping[edge_index[1]]
    edge_keep = (src_new >= 0) & (dst_new >= 0)
    edge_index_new = torch.stack([src_new[edge_keep], dst_new[edge_keep]], dim=0)
    edge_weight_new = edge_weight[edge_keep]
    n_id_new = n_id[keep]
    return n_id_new, edge_index_new, edge_weight_new, seed_size_new


class CombinedLoader:
    """
    Simple loader that concatenates multiple loaders each epoch.
    """

    def __init__(self, loaders: Sequence[Iterable[Dict[str, Any]]]) -> None:
        self.loaders = list(loaders)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for loader in self.loaders:
            for batch in loader:
                yield batch

    def __len__(self) -> int:
        total = 0
        for loader in self.loaders:
            if hasattr(loader, "__len__"):
                total += len(loader)  # type: ignore[arg-type]
        return total


class ShuffledCombinedLoader:
    """
    Memory-efficient joint shuffling across multiple loaders.

    Instead of materializing all batch objects (potentially very large), we only
    build a per-epoch schedule of loader indices and interleave batches by
    pulling from the corresponding underlying iterators.
    """

    def __init__(self, loaders: Sequence[Iterable[Dict[str, Any]]]) -> None:
        self.loaders = list(loaders)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if not self.loaders:
            return

        iters = [iter(loader) for loader in self.loaders]

        lengths: List[int] = []
        for loader in self.loaders:
            try:
                lengths.append(int(len(loader)))  # type: ignore[arg-type]
            except Exception:
                lengths = []
                break

        if lengths:
            schedule: List[int] = []
            for idx, n in enumerate(lengths):
                if n > 0:
                    schedule.extend([idx] * n)
            random.shuffle(schedule)

            for idx in schedule:
                try:
                    yield next(iters[idx])
                except StopIteration:
                    continue

            # If any iterator still has leftovers (shouldn't happen when `len()` is accurate),
            # exhaust them to avoid silently dropping data.
            for it in iters:
                for batch in it:
                    yield batch
            return

        # Fallback when lengths are unknown: randomly draw from active iterators.
        active = list(range(len(iters)))
        while active:
            pick = random.randrange(len(active))
            idx = active[pick]
            try:
                yield next(iters[idx])
            except StopIteration:
                active.pop(pick)

    def __len__(self) -> int:
        total = 0
        for loader in self.loaders:
            if hasattr(loader, "__len__"):
                total += len(loader)  # type: ignore[arg-type]
        return total


class BalancedShuffledCombinedLoader:
    """Interleave loaders while oversampling each one to the longest loader.

    GCAT builds separate ST and scRNA loaders.  Their natural concatenation
    makes the larger modality contribute more optimizer steps even though
    held-out ST genes receive direct expression supervision only from scRNA.
    This loader preserves every batch from the longer modality and cycles the
    shorter modality until both contribute the same number of batches.
    """

    def __init__(self, loaders: Sequence[Iterable[Dict[str, Any]]]) -> None:
        self.loaders = list(loaders)
        self.lengths = [int(len(loader)) for loader in self.loaders]
        self.target_length = max(self.lengths, default=0)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if not self.loaders or self.target_length <= 0:
            return

        schedule = [
            loader_idx
            for loader_idx in range(len(self.loaders))
            for _ in range(self.target_length)
        ]
        random.shuffle(schedule)
        iters = [iter(loader) for loader in self.loaders]
        for loader_idx in schedule:
            try:
                yield next(iters[loader_idx])
            except StopIteration:
                # Recreate only the exhausted shorter iterator.  NeighborLoader
                # reshuffles its input nodes, so repeated passes are not fixed
                # duplicate batches.
                iters[loader_idx] = iter(self.loaders[loader_idx])
                try:
                    yield next(iters[loader_idx])
                except StopIteration:
                    continue

    def __len__(self) -> int:
        return self.target_length * len(self.loaders)


class EmptyLoader:
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(())

    def __len__(self) -> int:
        return 0


def build_dataloaders(
    config: Union[OneHopLoaderConfig, Dict[str, Any], Any],
    use_gp: bool = True,
    order_required: Optional[bool] = None,
) -> Dict[str, Iterable[Dict[str, Any]]]:
    """
    Build one-hop neighbor dataloaders.

    Returns:
        {"train": train_loader, "val": val_loader, "test": test_loader}
    """
    # Backwards/ergonomic: allow passing `order_required` as a keyword argument
    # (in addition to putting it into `config`).
    if order_required is not None:
        if isinstance(config, dict):
            config = dict(config)
            config["order_required"] = bool(order_required)
        elif hasattr(config, "__dataclass_fields__"):
            config = dataclass_replace(config, order_required=bool(order_required))
        else:
            setattr(config, "order_required", bool(order_required))

    adata = _cfg_get(config, "adata")
    if adata is None:
        raise ValueError("`config` must provide an AnnData under key/attr `adata`.")

    split_field = _cfg_get(config, "split_field", "split")
    train_split = _cfg_get(config, "train_split", "train")
    val_split = _cfg_get(config, "val_split", "valid")
    test_split = _cfg_get(config, "test_split", "test")
    batch_size = int(_cfg_get(config, "batch_size", 128))
    num_neighbors = int(_cfg_get(config, "num_neighbors", 10))
    num_workers = int(_cfg_get(config, "num_workers", _cfg_get(config, "workers", 0)))
    shuffle_train = bool(_cfg_get(config, "shuffle_train", True))
    neighbor_agg = _cfg_get(config, "neighbor_agg", "mean")
    max_nodes_per_batch = _cfg_get(
        config, "max_nodes_per_batch", _cfg_get(config, "max_batch_nodes", None)
    )
    order_required = bool(_cfg_get(config, "order_required", False))
    balance_modalities = bool(_cfg_get(config, "balance_modalities", False))

    coords_key = _cfg_get(config, "coords_key", "spatial")
    coords_obs_cols = tuple(_cfg_get(config, "coords_obs_cols", ("x_FOV_px", "y_FOV_px")))
    spatial_adj_key = _cfg_get(config, "spatial_adj_key", None)
    spatial_k = int(_cfg_get(config, "spatial_k", 4))
    sc_knn_k = int(_cfg_get(config, "sc_knn_k", 10))
    sc_pca_n_components = int(_cfg_get(config, "sc_pca_n_components", 50))
    sc_log1p = bool(_cfg_get(config, "sc_log1p", True))

    covariate_fields = tuple(_cfg_get(config, "covariate_fields", ()))
    covariate_categories = _cfg_get(config, "covariate_categories", {})
    label_fields = tuple(_cfg_get(config, "label_fields", ()))
    observed_genes_cfg = _cfg_get(config, "observed_genes", None)
    gp_targets_mask_key = _cfg_get(config, "gp_targets_mask_key", "nichecompass_gp_targets")
    gp_sources_mask_key = _cfg_get(config, "gp_sources_mask_key", "nichecompass_gp_sources")
    gp_denovo_dim = int(_cfg_get(config, "gp_denovo_dim", 0))

    gene_list = list(adata.var_names)
    if isinstance(observed_genes_cfg, dict):
        observed_genes_st = (
            observed_genes_cfg.get("st")
            or observed_genes_cfg.get("spatial")
            or observed_genes_cfg.get("target")
        )
        observed_genes_sc = (
            observed_genes_cfg.get("sc")
            or observed_genes_cfg.get("nonspatial")
            or observed_genes_cfg.get("ref")
        )
    else:
        observed_genes_st = observed_genes_cfg
        observed_genes_sc = observed_genes_cfg

    gene_mask_st = _make_gene_mask_from_gene_list(gene_list, observed_genes_st)
    gene_mask_sc = _make_gene_mask_from_gene_list(gene_list, observed_genes_sc)

    # Detect spatial vs sc cells.
    spatial_mask: Optional[np.ndarray] = None
    if "platform" in adata.obs:
        spatial_mask = adata.obs["platform"].astype(str).str.lower().isin(SPATIAL_PLATFORM_LIST).to_numpy()
    else:
        coords_all = _coords_from_adata(adata, coords_key=coords_key, obs_cols=coords_obs_cols)
        if coords_all is not None:
            spatial_mask = ~np.any(np.isnan(coords_all), axis=1)

    if spatial_mask is None:
        spatial_mask = np.zeros(adata.n_obs, dtype=bool)

    has_spatial = bool(spatial_mask.any())
    has_sc = bool((~spatial_mask).any())

    loaders_train: List[Iterable[Dict[str, Any]]] = []
    loaders_val: List[Iterable[Dict[str, Any]]] = []
    loaders_test: List[Iterable[Dict[str, Any]]] = []

    def _split_indices(
        sub_obs_names: np.ndarray,
        split_value: Union[str, Sequence[str], None],
    ) -> Optional[np.ndarray]:
        if split_value is None:
            return None
        if split_field and split_field in adata.obs:
            split_series = adata.obs.loc[sub_obs_names, split_field].to_numpy()
            if isinstance(split_value, (list, tuple, set)):
                mask = np.isin(split_series, np.asarray(list(split_value)))
            else:
                mask = split_series == split_value
            idx = np.nonzero(mask)[0]
            return idx.astype(np.int64, copy=False) if idx.size > 0 else np.empty((0,), dtype=np.int64)
        return None

    # Build ST loaders (spatial coords)
    if has_spatial:
        sub_names = adata.obs_names[spatial_mask].to_numpy()
        adata_sub = adata[spatial_mask]
        global_node_ids = np.nonzero(spatial_mask)[0].astype(np.int64, copy=False)
        edge_index, edge_weight, coords = build_spatial_graph(
            adata_sub,
            k=spatial_k,
            coords_key=coords_key,
            obs_cols=coords_obs_cols,
            adj_key=spatial_adj_key,
            symmetrize=True,
        )
        # Map coords presence: even if missing, we still output coords from source
        coords_present = _coords_from_adata(adata_sub, coords_key=coords_key, obs_cols=coords_obs_cols) is not None

        train_nodes = _split_indices(sub_names, train_split)
        if train_nodes is None or train_nodes.size > 0:
            loaders_train.append(
                NCOneHopNeighborDataloader(
                    adata=adata_sub,
                    global_node_ids=global_node_ids,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    input_nodes=train_nodes,
                    gene_mask=gene_mask_st,
                    gene_list=gene_list,
                    split_field=split_field,
                    covariate_fields=covariate_fields,
                    covariate_categories=covariate_categories,
                    label_fields=label_fields,
                    neighbor_agg=neighbor_agg,
                    use_gp=use_gp,
                    gp_targets_mask_key=gp_targets_mask_key,
                    gp_sources_mask_key=gp_sources_mask_key,
                    gp_denovo_dim=gp_denovo_dim,
                    order_required=order_required,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    coords=coords,
                    coords_key_present=coords_present,
                    max_nodes_per_batch=max_nodes_per_batch,
                )
            )
        val_nodes = _split_indices(sub_names, val_split)
        if val_nodes is None or val_nodes.size > 0:
            loaders_val.append(
                NCOneHopNeighborDataloader(
                    adata=adata_sub,
                    global_node_ids=global_node_ids,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    input_nodes=val_nodes,
                    gene_mask=gene_mask_st,
                    gene_list=gene_list,
                    split_field=split_field,
                    covariate_fields=covariate_fields,
                    covariate_categories=covariate_categories,
                    label_fields=label_fields,
                    neighbor_agg=neighbor_agg,
                    use_gp=use_gp,
                    gp_targets_mask_key=gp_targets_mask_key,
                    gp_sources_mask_key=gp_sources_mask_key,
                    gp_denovo_dim=gp_denovo_dim,
                    order_required=order_required,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    shuffle=False,
                    num_workers=num_workers,
                    coords=coords,
                    coords_key_present=coords_present,
                    max_nodes_per_batch=max_nodes_per_batch,
                )
            )
        test_nodes = _split_indices(sub_names, test_split)
        if test_nodes is None or test_nodes.size > 0:
            loaders_test.append(
                NCOneHopNeighborDataloader(
                    adata=adata_sub,
                    global_node_ids=global_node_ids,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    input_nodes=test_nodes,
                    gene_mask=gene_mask_st,
                    gene_list=gene_list,
                    split_field=split_field,
                    covariate_fields=covariate_fields,
                    covariate_categories=covariate_categories,
                    label_fields=label_fields,
                    neighbor_agg=neighbor_agg,
                    use_gp=use_gp,
                    gp_targets_mask_key=gp_targets_mask_key,
                    gp_sources_mask_key=gp_sources_mask_key,
                    gp_denovo_dim=gp_denovo_dim,
                    order_required=order_required,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    shuffle=False,
                    num_workers=num_workers,
                    coords=coords,
                    coords_key_present=coords_present,
                    max_nodes_per_batch=max_nodes_per_batch,
                )
            )

    # Build sc loaders (PCA+KNN on expression)
    if has_sc:
        adata_sub = adata[~spatial_mask]
        sub_names = adata.obs_names[~spatial_mask].to_numpy()
        global_node_ids = np.nonzero(~spatial_mask)[0].astype(np.int64, copy=False)
        edge_index, edge_weight = build_expression_pca_knn_graph(
            adata_sub,
            k=sc_knn_k,
            n_components=sc_pca_n_components,
            log1p=sc_log1p,
            symmetrize=True,
        )
        train_nodes = _split_indices(sub_names, train_split)
        if train_nodes is None or train_nodes.size > 0:
            loaders_train.append(
                NCOneHopNeighborDataloader(
                    adata=adata_sub,
                    global_node_ids=global_node_ids,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    input_nodes=train_nodes,
                    gene_mask=gene_mask_sc,
                    gene_list=gene_list,
                    split_field=split_field,
                    covariate_fields=covariate_fields,
                    covariate_categories=covariate_categories,
                    label_fields=label_fields,
                    neighbor_agg=neighbor_agg,
                    use_gp=use_gp,
                    gp_targets_mask_key=gp_targets_mask_key,
                    gp_sources_mask_key=gp_sources_mask_key,
                    gp_denovo_dim=gp_denovo_dim,
                    order_required=order_required,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    coords=None,
                    coords_key_present=False,
                    max_nodes_per_batch=max_nodes_per_batch,
                )
            )
        val_nodes = _split_indices(sub_names, val_split)
        if val_nodes is None or val_nodes.size > 0:
            loaders_val.append(
                NCOneHopNeighborDataloader(
                    adata=adata_sub,
                    global_node_ids=global_node_ids,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    input_nodes=val_nodes,
                    gene_mask=gene_mask_sc,
                    gene_list=gene_list,
                    split_field=split_field,
                    covariate_fields=covariate_fields,
                    covariate_categories=covariate_categories,
                    label_fields=label_fields,
                    neighbor_agg=neighbor_agg,
                    use_gp=use_gp,
                    gp_targets_mask_key=gp_targets_mask_key,
                    gp_sources_mask_key=gp_sources_mask_key,
                    gp_denovo_dim=gp_denovo_dim,
                    order_required=order_required,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    shuffle=False,
                    num_workers=num_workers,
                    coords=None,
                    coords_key_present=False,
                    max_nodes_per_batch=max_nodes_per_batch,
                )
            )
        test_nodes = _split_indices(sub_names, test_split)
        if test_nodes is None or test_nodes.size > 0:
            loaders_test.append(
                NCOneHopNeighborDataloader(
                    adata=adata_sub,
                    global_node_ids=global_node_ids,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    input_nodes=test_nodes,
                    gene_mask=gene_mask_sc,
                    gene_list=gene_list,
                    split_field=split_field,
                    covariate_fields=covariate_fields,
                    covariate_categories=covariate_categories,
                    label_fields=label_fields,
                    neighbor_agg=neighbor_agg,
                    use_gp=use_gp,
                    gp_targets_mask_key=gp_targets_mask_key,
                    gp_sources_mask_key=gp_sources_mask_key,
                    gp_denovo_dim=gp_denovo_dim,
                    order_required=order_required,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    shuffle=False,
                    num_workers=num_workers,
                    coords=None,
                    coords_key_present=False,
                    max_nodes_per_batch=max_nodes_per_batch,
                )
            )

    train_loader: Iterable[Dict[str, Any]] = (
        BalancedShuffledCombinedLoader(loaders_train)
        if (
            len(loaders_train) > 1
            and shuffle_train
            and not order_required
            and balance_modalities
        )
        else ShuffledCombinedLoader(loaders_train)
        if (len(loaders_train) > 1 and shuffle_train and not order_required)
        else (
            CombinedLoader(loaders_train)
            if len(loaders_train) > 1
            else (loaders_train[0] if loaders_train else EmptyLoader())
        )
    )
    val_loader: Iterable[Dict[str, Any]] = (
        CombinedLoader(loaders_val)
        if len(loaders_val) > 1
        else (loaders_val[0] if loaders_val else EmptyLoader())
    )
    test_loader: Iterable[Dict[str, Any]] = (
        CombinedLoader(loaders_test)
        if len(loaders_test) > 1
        else (loaders_test[0] if loaders_test else EmptyLoader())
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
