"""规模档位：对齐 ``checkpoint/scaling.md`` 的 ``levels`` 模块。

``Level`` 只声明取值；阈值由配方的 scaling 阶梯给，代码里留一份默认表供
自检与样例使用。
"""

from __future__ import annotations

from typing import Literal, get_args

from .configs import Topology

Level = Literal["S0", "S1", "S2", "S3", "S4", "S5"]
LEVELS: tuple[Level, ...] = get_args(Level)

# 默认阶梯：总卡数落档。S0 由 nodes == 1 分支处理，不进这张表。
DEFAULT_LEVEL_BOUNDARIES: tuple[tuple[int, Level], ...] = (
    (8, "S1"),      # 8..255 卡
    (256, "S2"),    # 256..1023 卡
    (1024, "S3"),   # 1024..9999 卡
    (10000, "S4"),  # 10000 起
)


def resolve_level(topology: Topology, boundaries=None) -> Level:
    """按节点数与每节点卡数落到 S0 到 S5。

    单节点一律 S0；其余按 ``boundaries`` 的总卡数阈值落档，超过最后一档给 S5。
    """

    if topology.nodes == 1:
        # 单机最小参数规模是 S0；单机多卡属于 S1。
        return "S0" if topology.gpus_per_node <= 8 else "S1"
    if topology.nodes == 2:
        return "S1"

    total = topology.nodes * topology.gpus_per_node
    table = DEFAULT_LEVEL_BOUNDARIES if boundaries is None else boundaries
    for threshold, level in table:
        if total < threshold:
            return level
    return "S5"


def level_to_json(level: Level) -> str:
    return level


def level_from_json(raw: str) -> Level:
    if raw not in LEVELS:
        raise ValueError(f"未知规模档位: {raw!r}")
    return raw
