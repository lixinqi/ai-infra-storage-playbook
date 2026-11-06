"""checkpoint 样本接口的 public 层。

只做转发：从 ``ckpt_sample`` 与 ``ckpt_sample_api`` 里 import，再把要发布的名字
重新绑定一次，对齐 ``checkpoint/ckpt-sample-interface.md`` 的 ``public`` 模块。
实现类走各自的子模块（``sample.memory``、``sample.types``、``sample.configs``、
``sample.levels``）。
"""

from . import ckpt_sample, ckpt_sample_api

Start = ckpt_sample_api.Start
Stop = ckpt_sample_api.Stop
Query = ckpt_sample_api.Query
Execute = ckpt_sample_api.Execute
GetLevel = ckpt_sample_api.GetLevel

RunSpec = ckpt_sample.RunSpec
RunRecord = ckpt_sample.RunRecord
Artifacts = ckpt_sample.Artifacts
CkptSli = ckpt_sample.CkptSli
CkptSlo = ckpt_sample.CkptSlo
SampleConfigBase = ckpt_sample.SampleConfigBase

__all__ = [
    "ckpt_sample", "ckpt_sample_api",
    "Start", "Stop", "Query", "Execute", "GetLevel",
    "RunSpec", "RunRecord", "Artifacts", "CkptSli", "CkptSlo",
    "SampleConfigBase",
]
