"""Random helper functions without another home for now."""

from copy import deepcopy
from typing import Any

import numpy as np
from qcdata import Structure


def randomly_reorder_structure(struct: "Structure") -> "Structure":
    n = len(struct.symbols)
    perm = np.random.permutation(n).astype(int)  # new_index -> old_index

    # inverse: old_index -> new_index
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n, dtype=int)

    d: dict[str, Any] = deepcopy(struct.model_dump())

    # Per-atom fields (add others your graph/coloring uses)
    d["symbols"] = [struct.symbols[old] for old in perm]
    d["geometry"] = np.asarray(struct.geometry)[perm]

    # Reindex connectivity triplets; normalize endpoints so i <= j
    if getattr(struct, "connectivity", None) is not None:
        new_conn = []
        for i_old, j_old, order in struct.connectivity:
            i_new = int(inv[int(i_old)])
            j_new = int(inv[int(j_old)])
            if i_new > j_new:
                i_new, j_new = j_new, i_new
            new_conn.append((i_new, j_new, float(order)))
        d["connectivity"] = new_conn

    return Structure.model_validate(d)
