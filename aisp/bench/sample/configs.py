"""样本配置基类。

对齐 ``checkpoint/ckpt-sample-interface.md`` 的 ``ckpt_sample`` 模块。
样本 repo 里每份配置必须包含这三个子结构；样本可以在此基础上扩展自己的字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from .observation import ObservationLayer
from .types import _only


@dataclass(frozen=True)
class Model:
    id: str         # HuggingFace repo id 之类
    git_commit_id: str

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "git_commit_id": self.git_commit_id}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Model":
        _only(raw, ('id', 'git_commit_id'), "Model")
        return cls(id=raw["id"], git_commit_id=raw["git_commit_id"])


Network = Literal["rdma", "tcp", "none"]
Mount = Literal["local-nvme", "nfs", "fuse"]


@dataclass(frozen=True)
class Topology:
    nodes: int
    gpus_per_node: int
    network: Network
    mount: Mount

    def __post_init__(self) -> None:
        # schema 只写了 integer，没给下限；0 节点的拓扑算出来的档位没有意义，
        # 样本侧也会退化成空跑，这里直接拒绝。
        if self.nodes < 1:
            raise ValueError(f"nodes 至少为 1，得到 {self.nodes}")
        if self.gpus_per_node < 1:
            raise ValueError(f"gpus_per_node 至少为 1，得到 {self.gpus_per_node}")

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "network": self.network,
            "mount": self.mount,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Topology":
        _only(raw, ('nodes', 'gpus_per_node', 'network', 'mount'), "Topology")
        return cls(
            nodes=raw["nodes"],
            gpus_per_node=raw["gpus_per_node"],
            network=cast(Network, raw["network"]),
            mount=cast(Mount, raw["mount"]),
        )


@dataclass(frozen=True)
class ObservationConfig:
    enabled: bool
    sample_rate: float  # 0..1
    layers: tuple[ObservationLayer, ...] = ()

    def __post_init__(self) -> None:
        # 文档：采集层至少两层，其中必须含 sys（应用层与系统层的总字节数要交叉核对）。
        if self.enabled:
            if len(self.layers) < 2:
                raise ValueError(f"采集层至少两层，得到 {list(self.layers)}")
            if "sys" not in self.layers:
                raise ValueError(f"采集层必须含 sys，得到 {list(self.layers)}")

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "layers": list(self.layers),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ObservationConfig":
        _only(raw, ('enabled', 'sample_rate', 'layers'), "ObservationConfig")
        return cls(
            enabled=raw["enabled"],
            sample_rate=raw["sample_rate"],
            layers=tuple(cast(ObservationLayer, item) for item in raw["layers"]),
        )


@dataclass(frozen=True)
class SampleConfigBase:
    """样本 repo 每份配置必须含有的基类字段。"""

    model: Model
    topology: Topology
    observe: ObservationConfig

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model.to_json(),
            "topology": self.topology.to_json(),
            "observe": self.observe.to_json(),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "SampleConfigBase":
        _only(raw, ('model', 'topology', 'observe'), "SampleConfigBase")
        return cls(
            model=Model.from_json(raw["model"]),
            topology=Topology.from_json(raw["topology"]),
            observe=ObservationConfig.from_json(raw["observe"]),
        )


# 校验用：合法枚举值集合。
NETWORK_VALUES = set(get_args(Network))
MOUNT_VALUES = set(get_args(Mount))
OBSERVATION_LAYER_VALUES = set(get_args(ObservationLayer))
