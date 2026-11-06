"""SampleService 的内存参考实现。

只用于开发期自检与单机回放；不做进程隔离、不落盘到对象存储。
Start 在后台线程里跑 Execute，Query/Stop 通过锁与事件同步状态。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace
from typing import Callable, Dict, Optional

from .ckpt_sample_api import SampleRunner
from .configs import SampleConfigBase
from .levels import Level, resolve_level
from .types import (
    Artifacts,
    Cancelled,
    Completed,
    Failed,
    Failure,
    Queued,
    QueueProgress,
    Rejected,
    Running,
    RunRecord,
    RunSpec,
    State,
    WorkProgress,
)


ConfigLoader = Callable[[str], SampleConfigBase]
ArtifactWriter = Callable[[str, RunSpec, SampleConfigBase, Artifacts], Artifacts]

# 终态：到达后不再接受状态变更，stop() 也原样交回。
_TERMINAL = ("completed", "failed", "cancelled", "rejected")


def _default_artifact_writer(
    run_id: str, spec: RunSpec, config: SampleConfigBase, artifacts: Artifacts
) -> Artifacts:
    """默认 writer：只把路径写成内存占位，不落盘。生产实现替换成对象存储。"""

    return replace(
        artifacts,
        manifest=f"memory://runs/{run_id}/manifest.json",
        report=f"memory://runs/{run_id}/report.html",
    )


class InMemorySampleService:
    """单机同步/异步参考实现。

    - ``runners``：按 ``sample`` 字段（``name@version``）注册的样本执行器。
    - ``config_loader``：按 spec.config 名字加载 SampleConfigBase。
    - ``artifact_writer``：Execute 结束后落盘四份产物，返回带路径的 Artifacts。
    """

    def __init__(
        self,
        runners: Dict[str, SampleRunner],
        config_loader: ConfigLoader,
        artifact_writer: Optional[ArtifactWriter] = None,
        max_records: int = 1000,
    ) -> None:
        self._runners = dict(runners)
        self._load_config = config_loader
        self._write_artifacts = artifact_writer or _default_artifact_writer
        # 参考实现把记录留在内存里供 Query 回查；超过上限先淘汰最早的终态记录。
        self._max_records = max(1, max_records)

        self._lock = threading.Lock()
        self._records: Dict[str, RunRecord] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------
    # SampleService
    # ------------------------------------------------------------------

    def get_level(self, config: SampleConfigBase) -> Level:
        return resolve_level(config.topology)

    def start(self, spec: RunSpec) -> RunRecord:
        run_id = uuid.uuid4().hex
        try:
            self._validate(spec)
            # config 也要在 start 时解析掉：加载不出来的请求没资格进队列，
            # 应该直接 Rejected，而不是起线程后再 Failed。
            config = self._load_config(spec.config)
        except ValueError as exc:
            record = RunRecord(
                run_id=run_id,
                state=Rejected(error=Failure(kind="request", detail=str(exc))),
            )
            with self._lock:
                self._records[run_id] = record
                self._evict_terminal_locked()
            return record
        except Exception as exc:  # noqa: BLE001 - config 加载异常同样归为请求不合法
            record = RunRecord(
                run_id=run_id,
                state=Rejected(error=Failure(kind="request", detail=f"config: {type(exc).__name__}: {exc}")),
            )
            with self._lock:
                self._records[run_id] = record
                self._evict_terminal_locked()
            return record

        with self._lock:
            record = RunRecord(
                run_id=run_id,
                # 内存实现没有等待队列：start 后立即起线程。pos/len 恒为 0，
                # 表示「刚登记、不排队」，不是编造一个排队位置。
                state=Queued(queue=QueueProgress(pos=0, len=0)),
                progress=WorkProgress(0, 0),
            )
            self._records[run_id] = record
            cancel_event = threading.Event()
            self._cancel_events[run_id] = cancel_event

        thread = threading.Thread(
            target=self._run_background,
            args=(run_id, spec, config, cancel_event),
            daemon=True,
        )
        self._threads[run_id] = thread
        thread.start()
        return record

    def stop(self, run_id: str) -> RunRecord:
        """取消一次尚未结束的运行；返回时状态已是 cancelled。

        取消是请求，不是等待：在锁内立刻把状态翻成 cancelled 再返回，
        不把「后台线程还没响应」暴露成 running。样本可能在下一个检查点才停，
        但对外状态从这一刻起就是 cancelled。
        """

        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise KeyError(run_id)
            if record.state.tag in _TERMINAL:
                return record
            event = self._cancel_events.get(run_id)
            cancelled = RunRecord(
                run_id=run_id,
                state=Cancelled(),
                progress=record.progress,
                artifacts=record.artifacts,
            )
            self._records[run_id] = cancelled

        if event is not None:
            event.set()
        return cancelled

    def query(self, run_id: str) -> RunRecord:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return record

    def execute(self, spec: RunSpec) -> Artifacts:
        self._validate(spec)
        config = self._load_config(spec.config)
        runner = self._runners[spec.sample]
        metrics = list(runner(spec, config))
        artifacts = Artifacts(metrics=tuple(metrics), slo=())
        # 同步执行没有对外运行标识；给 writer 一个不含 @ 与 . 的局部标识。
        return self._write_artifacts(f"sync-{uuid.uuid4().hex}", spec, config, artifacts)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _validate(self, spec: RunSpec) -> None:
        if "@" not in spec.sample:
            raise ValueError(f"sample 必须写成 name@version，得到 {spec.sample!r}")
        if spec.sample not in self._runners:
            raise ValueError(f"未注册的样本: {spec.sample}")
        if not isinstance(spec.seed, int) or isinstance(spec.seed, bool) or spec.seed < 0:
            raise ValueError(f"seed 必须是非负整数，得到 {spec.seed!r}")
        for layer in (spec.storage.backend, spec.storage.adapter):
            if not layer.name.strip() or not layer.version.strip():
                raise ValueError(f"组件 name/version 不能为空: {layer}")

    def _run_background(
        self,
        run_id: str,
        spec: RunSpec,
        config: SampleConfigBase,
        cancel_event: threading.Event,
    ) -> None:
        try:
            self._set_state(run_id, Running(), progress=WorkProgress(0, 0))
            if cancel_event.is_set():
                self._set_state(run_id, Cancelled())
                return

            runner = self._runners[spec.sample]

            # 样本执行期间，runner 自己决定是否响应取消；这里只在结束点检查一次。
            metrics = list(runner(spec, config))
            if cancel_event.is_set():
                self._set_state(run_id, Cancelled())
                return

            artifacts = self._write_artifacts(run_id, spec, config, Artifacts(metrics=tuple(metrics)))
            self._set_state(
                run_id,
                Completed(),
                # 参考实现不报告细粒度工作量，保持 0/0（未知）；完成与否看状态。
                # 用 SLI 条数冒充 total 是错把结果当进度。
                progress=WorkProgress(0, 0),
                artifacts=artifacts,
            )
        except TimeoutError as exc:
            self._set_state(run_id, Failed(error=Failure(kind="timeout", detail=str(exc))))
        except Exception as exc:  # noqa: BLE001 - 其余异常无法细分，归为 assertion
            self._set_state(
                run_id,
                Failed(error=Failure(kind="assertion", detail=f"{type(exc).__name__}: {exc}")),
            )
        finally:
            with self._lock:
                self._threads.pop(run_id, None)
                self._cancel_events.pop(run_id, None)

    def _set_state(
        self,
        run_id: str,
        state: State,
        progress: Optional[WorkProgress] = None,
        artifacts: Optional[Artifacts] = None,
    ) -> None:
        with self._lock:
            old = self._records[run_id]
            # cancelled 是吸收态：stop() 之后后台线程迟到的 running / completed /
            # failed 都不许把取消结果覆盖掉。
            if old.state.tag == "cancelled":
                return
            self._records[run_id] = RunRecord(
                run_id=run_id,
                state=state,
                progress=progress if progress is not None else old.progress,
                artifacts=artifacts if artifacts is not None else old.artifacts,
            )
            if state.tag in _TERMINAL:
                self._evict_terminal_locked()

    def _evict_terminal_locked(self) -> None:
        """超过 max_records 时按插入顺序淘汰最早的终态记录；运行中的不动。"""

        if len(self._records) <= self._max_records:
            return
        for run_id, record in list(self._records.items()):
            if len(self._records) <= self._max_records:
                return
            if record.state.tag in _TERMINAL:
                del self._records[run_id]
