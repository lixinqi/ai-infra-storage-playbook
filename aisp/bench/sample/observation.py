"""观测采集层：对齐 ``checkpoint/methodology.md`` 的 ``observation`` 模块。"""

from __future__ import annotations

from typing import Literal, get_args

ObservationLayer = Literal["app", "sys", "storage"]
OBSERVATION_LAYERS: tuple[ObservationLayer, ...] = get_args(ObservationLayer)
