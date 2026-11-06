"""Content fingerprints for CCC inputs; never reuse a graph across holdout masks."""

import hashlib
import json

import numpy as np
from scipy import sparse


def content_digest(*values):
    digest = hashlib.sha256()
    for value in values:
        if sparse.issparse(value):
            value = value.tocsr()
            parts = (np.asarray(value.shape), value.indptr, value.indices, value.data)
        elif isinstance(value, np.ndarray):
            parts = (value,)
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
            continue
        for part in parts:
            digest.update(str(part.dtype).encode())
            digest.update(str(part.shape).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


def input_signature(adata, lr_programs, topk, candidate_graph):
    return content_digest(
        list(map(str, adata.obs_names)), list(map(str, adata.var_names)),
        adata.X, lr_programs, int(topk), candidate_graph,
    )
