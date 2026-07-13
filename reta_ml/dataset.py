"""
PyTorch Geometric Dataset for RETA field graphs.

Each sample is one field, stored as a ``torch_geometric.data.Data`` object
with three edge-index attributes (temporal, spatial, transect) rather than
the standard single ``edge_index``.  This keeps full compatibility with PyG
batching (``DataLoader`` / ``Batch``), which auto-increments all
``*edge_index*`` tensors.

Sprint 4 introduces a ``HeteroData`` view of the same graph: a single node
type (``"point"``) with three edge types (``"temporal"``, ``"spatial"``,
``"transect"``).  This enables joint message passing with ``HeteroConv`` while
reusing the Sprint 1–3 pipeline unchanged.

**Important**: augmentation must be applied **before** graph construction so
that edge topology matches the (possibly transformed) coordinates.  Train
graphs are built on augmented coordinates; test/val graphs on originals.
"""

from typing import List, Optional, Sequence

import pandas as pd

try:
    import torch
    from torch_geometric.data import Data, Dataset, HeteroData
except ImportError:
    raise ImportError(
        "reta_ml.dataset requires PyTorch and PyTorch Geometric. "
        "Install with: pip install torch torch_geometric"
    )

from reta_ml.graph import build_field_graph


# ---------------------------------------------------------------------------
# Single-graph helper
# ---------------------------------------------------------------------------

def field_df_to_data(
    df: pd.DataFrame,
    *,
    temporal_K: int = 3,
    spatial_R: float = 15.0,
    spatial_max_neighbors: Optional[int] = None,
    transect_K: int = 1,
    time_gap_threshold: Optional[float] = None,
    time_gap_n_sigma: float = 3.0,
    main_variable: Optional[str] = None,
) -> "Data":
    """Convert a single-field preprocessed DataFrame into a PyG ``Data``.

    The returned object carries:

    * ``x``  – node feature matrix  ``[N, F]``
    * ``y``  – integer class labels  ``[N]``
    * ``edge_index_temporal``  – ``[2, E_t]``
    * ``edge_index_spatial``   – ``[2, E_s]``
    * ``edge_index_transect``  – ``[2, E_tr]``
    * ``num_nodes``

    All ``edge_index_*`` attributes are automatically incremented by PyG's
    ``Batch.from_data_list`` because they match the ``edge_index`` naming
    pattern.
    """
    g = build_field_graph(
        df,
        temporal_K=temporal_K,
        spatial_R=spatial_R,
        spatial_max_neighbors=spatial_max_neighbors,
        transect_K=transect_K,
        time_gap_threshold=time_gap_threshold,
        time_gap_n_sigma=time_gap_n_sigma,
        main_variable=main_variable,
    )

    data = Data(
        x=torch.from_numpy(g["x"]),
        y=torch.from_numpy(g["y"]),
        edge_index_temporal=torch.from_numpy(g["edge_index_temporal"]),
        edge_index_spatial=torch.from_numpy(g["edge_index_spatial"]),
        edge_index_transect=torch.from_numpy(g["edge_index_transect"]),
        edge_attr_temporal=torch.from_numpy(g["edge_attr_temporal"]),
        edge_attr_spatial=torch.from_numpy(g["edge_attr_spatial"]),
        edge_attr_transect=torch.from_numpy(g["edge_attr_transect"]),
        num_nodes=g["num_nodes"],
    )
    return data


# ---------------------------------------------------------------------------
# HeteroData helper (Sprint 4)
# ---------------------------------------------------------------------------

EDGE_TYPE_TEMPORAL = ("point", "temporal", "point")
EDGE_TYPE_SPATIAL = ("point", "spatial", "point")
EDGE_TYPE_TRANSECT = ("point", "transect", "point")


def data_to_heterodata(data: "Data") -> "HeteroData":
    """Convert a Sprint 3 ``Data`` object into Sprint 4 ``HeteroData``.

    Node type: ``"point"``.
    Edge types: ``("point","temporal","point")``, ``("point","spatial","point")``,
    ``("point","transect","point")``.
    """
    hd = HeteroData()
    hd["point"].x = data.x
    if hasattr(data, "y") and data.y is not None:
        hd["point"].y = data.y

    hd[EDGE_TYPE_TEMPORAL].edge_index = data.edge_index_temporal
    hd[EDGE_TYPE_SPATIAL].edge_index = data.edge_index_spatial
    hd[EDGE_TYPE_TRANSECT].edge_index = data.edge_index_transect

    if getattr(data, "edge_attr_temporal", None) is not None:
        hd[EDGE_TYPE_TEMPORAL].edge_attr = data.edge_attr_temporal
    if getattr(data, "edge_attr_spatial", None) is not None:
        hd[EDGE_TYPE_SPATIAL].edge_attr = data.edge_attr_spatial
    if getattr(data, "edge_attr_transect", None) is not None:
        hd[EDGE_TYPE_TRANSECT].edge_attr = data.edge_attr_transect

    if getattr(data, "num_nodes", None) is not None:
        hd["point"].num_nodes = int(data.num_nodes)

    return hd


def field_df_to_heterodata(
    df: pd.DataFrame,
    *,
    temporal_K: int = 3,
    spatial_R: float = 15.0,
    spatial_max_neighbors: Optional[int] = None,
    transect_K: int = 1,
    time_gap_threshold: Optional[float] = None,
    time_gap_n_sigma: float = 3.0,
    main_variable: Optional[str] = None,
) -> "HeteroData":
    """Convert a single-field DataFrame into a ``HeteroData`` graph."""
    d = field_df_to_data(
        df,
        temporal_K=temporal_K,
        spatial_R=spatial_R,
        spatial_max_neighbors=spatial_max_neighbors,
        transect_K=transect_K,
        time_gap_threshold=time_gap_threshold,
        time_gap_n_sigma=time_gap_n_sigma,
        main_variable=main_variable,
    )
    return data_to_heterodata(d)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RETAFieldDataset(Dataset):
    """PyG ``Dataset`` wrapping one or more preprocessed field tables.

    Graphs are built eagerly at construction time and held in memory.
    Compatible with ``torch_geometric.loader.DataLoader`` for batching
    multiple fields into a single disconnected graph via ``Batch``.

    Parameters
    ----------
    field_dfs : list[pd.DataFrame]
        One DataFrame per field (already preprocessed, split, and optionally
        augmented).  Each DataFrame becomes one ``Data`` graph.
    temporal_K, spatial_R, transect_K, time_gap_threshold, time_gap_n_sigma
        Graph construction hyper-parameters (forwarded to
        :func:`field_df_to_data`).
    main_variable : str, optional
        Main variable column name.
    """

    def __init__(
        self,
        field_dfs: Sequence[pd.DataFrame],
        *,
        temporal_K: int = 3,
        spatial_R: float = 15.0,
        spatial_max_neighbors: Optional[int] = None,
        transect_K: int = 1,
        time_gap_threshold: Optional[float] = None,
        time_gap_n_sigma: float = 3.0,
        main_variable: Optional[str] = None,
    ):
        super().__init__()

        self._graphs: List[Data] = []
        for df in field_dfs:
            d = field_df_to_data(
                df,
                temporal_K=temporal_K,
                spatial_R=spatial_R,
                spatial_max_neighbors=spatial_max_neighbors,
                transect_K=transect_K,
                time_gap_threshold=time_gap_threshold,
                time_gap_n_sigma=time_gap_n_sigma,
                main_variable=main_variable,
            )
            self._graphs.append(d)

    # Required overrides ----------------------------------------------------

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> "Data":
        return self._graphs[idx]


class RETAHeteroFieldDataset(Dataset):
    """PyG ``Dataset`` returning ``HeteroData`` graphs (Sprint 4).

    Mirrors :class:`RETAFieldDataset` but converts each field graph to a
    ``HeteroData`` view with three edge types.
    """

    def __init__(
        self,
        field_dfs: Sequence[pd.DataFrame],
        *,
        temporal_K: int = 3,
        spatial_R: float = 15.0,
        spatial_max_neighbors: Optional[int] = None,
        transect_K: int = 1,
        time_gap_threshold: Optional[float] = None,
        time_gap_n_sigma: float = 3.0,
        main_variable: Optional[str] = None,
    ):
        super().__init__()

        self._graphs: List[HeteroData] = []
        for df in field_dfs:
            hd = field_df_to_heterodata(
                df,
                temporal_K=temporal_K,
                spatial_R=spatial_R,
                spatial_max_neighbors=spatial_max_neighbors,
                transect_K=transect_K,
                time_gap_threshold=time_gap_threshold,
                time_gap_n_sigma=time_gap_n_sigma,
                main_variable=main_variable,
            )
            self._graphs.append(hd)

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> "HeteroData":
        return self._graphs[idx]
