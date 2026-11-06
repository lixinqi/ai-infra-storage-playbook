"""五个接口的签名。

对齐 ``checkpoint/ckpt-sample-interface.md`` 的 ``ckpt_sample_api`` 模块。
调用方只依赖这个 Protocol；内存实现与后续 HTTP/CLI 适配都实现它。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .ckpt_sample import Artifacts, RunRecord, RunSpec, SampleConfigBase
from .levels import Level


@runtime_checkable
class SampleService(Protocol):
    """一份 RunSpec 进，一条 RunRecord 出。"""

    def start(self, spec: RunSpec) -> RunRecord:
        """提交接口：校验后入队并在后台执行，立即返回；不等样本跑完。"""
        ...

    def stop(self, run_id: str) -> RunRecord:
        """取消接口：让这次运行停下，已写完的产物保留。已结束的运行原样返回。"""
        ...

    def query(self, run_id: str) -> RunRecord:
        """查一次运行的当前状态、进度与产物位置。只读。"""
        ...

    def execute(self, spec: RunSpec) -> Artifacts:
        """执行接口：同一份请求同步跑到结束，返回时四份产物已写完。"""
        ...

    def get_level(self, config: SampleConfigBase) -> Level:
        """导出属性：按节点数与每节点卡数落到 S0..S5。"""
        ...


@runtime_checkable
class SampleRunner(Protocol):
    """样本程序作者实现的 callable：给定 spec 与配置，产出本次运行的 SLI。

    服务在 ``execute`` 里调用它；产物落盘由服务完成。
    """

    def __call__(self, spec: RunSpec, config: SampleConfigBase) -> list:  # list[CkptSli]
        ...


# 文档把五个接口当独立名字导出；这里用未绑定方法表达，public 层直接转发。
Start = SampleService.start
Stop = SampleService.stop
Query = SampleService.query
Execute = SampleService.execute
GetLevel = SampleService.get_level
