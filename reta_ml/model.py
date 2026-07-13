"""
Sprint 4 models for RETA ML.

- HeteroGAT: joint message passing over three edge types using HeteroConv.
- ThreeGNNEnsemble: separate GAT stacks per edge type, logits combined at end.

Both operate on a single-node-type PyG HeteroData graph:
  node type: "point"
  edge types: ("point","temporal","point"), ("point","spatial","point"),
              ("point","transect","point")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATConv, HeteroConv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "reta_ml.model requires torch_geometric. Install with: pip install torch_geometric"
    ) from e


NodeType = Literal["point"]
EdgeType = Tuple[NodeType, Literal["temporal", "spatial", "transect"], NodeType]

EDGE_TYPES: Tuple[EdgeType, ...] = (
    ("point", "temporal", "point"),
    ("point", "spatial", "point"),
    ("point", "transect", "point"),
)


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 96
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    conv_aggr: Literal["sum", "mean", "min", "max", "mul"] = "sum"
    residual: bool = True


def _ensure_supported_layers(num_layers: int) -> None:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if num_layers > 8:
        raise ValueError("num_layers > 8 is not supported for this project")


class HeteroGAT(nn.Module):
    """HeteroGAT (single graph, 3 edge types, fused each layer)."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 96,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        conv_aggr: Literal["sum", "mean", "min", "max", "mul"] = "sum",
        residual: bool = True,
        num_classes: int = 4,
        edge_dim: Optional[int] = 2,
    ):
        super().__init__()
        _ensure_supported_layers(num_layers)
        if heads < 1:
            raise ValueError("heads must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.conv_aggr = conv_aggr
        self.residual = bool(residual)
        self.num_classes = int(num_classes)
        self.edge_dim = int(edge_dim) if edge_dim else None

        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            convs = {
                et: GATConv(
                    (-1, -1),
                    self.hidden_dim,
                    heads=self.heads,
                    concat=False,  # keep dim = hidden_dim regardless of heads
                    dropout=self.dropout,
                    add_self_loops=False,
                    edge_dim=self.edge_dim,  # distance/bearing-aware attention
                )
                for et in EDGE_TYPES
            }
            self.layers.append(HeteroConv(convs, aggr=self.conv_aggr))

        self.act = nn.ELU()
        self.drop = nn.Dropout(self.dropout)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, data: "HeteroData") -> Tensor:
        x = data["point"].x
        x = self.drop(self.act(self.input_proj(x)))

        # Pass edge attributes when present so attention is distance-aware.
        edge_attr_dict = None
        if self.edge_dim is not None:
            try:
                ea = data.edge_attr_dict
                if ea and all(v is not None for v in ea.values()):
                    edge_attr_dict = ea
            except (AttributeError, KeyError):
                edge_attr_dict = None

        x_dict: Dict[str, Tensor] = {"point": x}
        for conv in self.layers:
            if edge_attr_dict is not None:
                out_dict = conv(x_dict, data.edge_index_dict,
                                edge_attr_dict=edge_attr_dict)
            else:
                out_dict = conv(x_dict, data.edge_index_dict)
            out = out_dict["point"]
            out = self.drop(self.act(out))
            if self.residual and out.shape == x_dict["point"].shape:
                x_dict["point"] = x_dict["point"] + out
            else:
                x_dict["point"] = out

        logits = self.head(x_dict["point"])
        return logits


class _GATStack(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        num_classes: int,
        edge_dim: Optional[int] = 2,
    ):
        super().__init__()
        _ensure_supported_layers(num_layers)

        self.edge_dim = int(edge_dim) if edge_dim else None
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                GATConv(
                    (-1, -1),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=self.edge_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        h = self.drop(self.act(self.input_proj(x)))
        use_ea = self.edge_dim is not None and edge_attr is not None
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_attr) if use_ea else conv(h, edge_index)
            h = self.drop(self.act(h))
        return self.head(h)


class ThreeGNNEnsemble(nn.Module):
    """Ablation: 3 independent GATs (one per edge type), combine logits."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 96,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 4,
        combine: Literal["mean", "sum", "learned"] = "learned",
    ):
        super().__init__()
        if combine not in ("mean", "sum", "learned"):
            raise ValueError("combine must be one of: mean, sum, learned")

        self.temporal = _GATStack(
            in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.spatial = _GATStack(
            in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.transect = _GATStack(
            in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.combine = combine
        if self.combine == "learned":
            self.logit_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        else:
            self.register_parameter("logit_weights", None)

    def forward(self, data: "HeteroData") -> Tensor:
        x = data["point"].x

        def _ea(et):
            store = data[et]
            return getattr(store, "edge_attr", None)

        et_t = ("point", "temporal", "point")
        et_s = ("point", "spatial", "point")
        et_tr = ("point", "transect", "point")
        lt = self.temporal(x, data[et_t].edge_index, _ea(et_t))
        ls = self.spatial(x, data[et_s].edge_index, _ea(et_s))
        ltr = self.transect(x, data[et_tr].edge_index, _ea(et_tr))

        if self.combine == "sum":
            return lt + ls + ltr
        if self.combine == "mean":
            return (lt + ls + ltr) / 3.0

        w = torch.softmax(self.logit_weights, dim=0)
        return w[0] * lt + w[1] * ls + w[2] * ltr


def model_from_name(
    name: str,
    *,
    in_dim: int,
    cfg: Optional[ModelConfig] = None,
) -> nn.Module:
    cfg = cfg or ModelConfig()
    name = name.lower().strip()
    if name in ("heterogat", "hetero_gat", "hetero"):
        return HeteroGAT(
            in_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.heads,
            dropout=cfg.dropout,
            conv_aggr=cfg.conv_aggr,
            residual=cfg.residual,
        )
    if name in ("ensemble", "three_gnn", "three-gnn", "threegnn"):
        return ThreeGNNEnsemble(
            in_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.heads,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model name: {name}")

