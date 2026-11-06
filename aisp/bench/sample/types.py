"""Checkpoint 样本接口的类型定义。

对齐 ``checkpoint/ckpt-sample-interface.md`` 的 ``ckpt_sample`` 模块，
序列化结果与 ``checkpoint/json_schema/`` 的六份 schema 逐字段一致。

oneof 在 JSON 里是单键对象：``{"queued": {...}}``、``{"running": null}``、
``{"check_point_duration_sli": 0.064}``。dataclass 用 ``tag`` 字段承载键名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Optional, Union

# ---------------------------------------------------------------------------
# 存储两层
# ---------------------------------------------------------------------------


def _only(raw: dict[str, Any], allowed: tuple[str, ...], what: str) -> None:
    """JSON Schema 用 additionalProperties: false；解码时同样拒绝多余键。

    from_json 宽松接受未知键会让「schema 校验过的载荷」和「解码器接受的载荷」
    出现两套口径，这里对齐成一套。
    """

    extra = set(raw) - set(allowed)
    if extra:
        raise ValueError(f"{what} 含多余的键: {sorted(extra)}")


@dataclass(frozen=True)
class Component:
    """一个存储组件：纯存储系统或传输适配层。"""

    name: str
    version: str
    git_commit_id: Optional[str] = None  # 只发版本不发源码时为 None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "git_commit_id": self.git_commit_id,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Component":
        _only(raw, ('name', 'version', 'git_commit_id'), "Component")
        return cls(
            name=raw["name"],
            version=raw["version"],
            git_commit_id=raw["git_commit_id"],
        )


@dataclass(frozen=True)
class StorageLayer:
    """一次运行要评的存储两层：backend 纯存储，adapter 传输适配。"""

    backend: Component
    adapter: Component

    def to_json(self) -> dict[str, Any]:
        return {"backend": self.backend.to_json(), "adapter": self.adapter.to_json()}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "StorageLayer":
        _only(raw, ('backend', 'adapter'), "StorageLayer")
        return cls(
            backend=Component.from_json(raw["backend"]),
            adapter=Component.from_json(raw["adapter"]),
        )


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueProgress:
    pos: int
    len: int

    def to_json(self) -> dict[str, Any]:
        return {"pos": self.pos, "len": self.len}


@dataclass(frozen=True)
class WorkProgress:
    done: int
    total: int  # 还没算出来时为 0

    def to_json(self) -> dict[str, Any]:
        return {"done": self.done, "total": self.total}


@dataclass(frozen=True)
class Failure:
    kind: str  # environment / dependency / timeout / observation / assertion ...
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Failure":
        _only(raw, ('kind', 'detail'), "Failure")
        return cls(kind=raw["kind"], detail=raw["detail"])


@dataclass(frozen=True)
class Queued:
    tag: ClassVar[str] = "queued"
    queue: QueueProgress

    def to_json(self) -> dict[str, Any]:
        return {"queued": self.queue.to_json()}


@dataclass(frozen=True)
class Running:
    tag: ClassVar[str] = "running"

    def to_json(self) -> dict[str, Any]:
        return {"running": None}


@dataclass(frozen=True)
class Completed:
    tag: ClassVar[str] = "completed"

    def to_json(self) -> dict[str, Any]:
        return {"completed": None}


@dataclass(frozen=True)
class Failed:
    tag: ClassVar[str] = "failed"
    error: Failure = field(default_factory=lambda: Failure(kind="assertion", detail=""))

    def to_json(self) -> dict[str, Any]:
        return {"failed": self.error.to_json()}


@dataclass(frozen=True)
class Rejected:
    tag: ClassVar[str] = "rejected"
    error: Failure = field(default_factory=lambda: Failure(kind="assertion", detail=""))

    def to_json(self) -> dict[str, Any]:
        return {"rejected": self.error.to_json()}


@dataclass(frozen=True)
class Cancelled:
    tag: ClassVar[str] = "cancelled"

    def to_json(self) -> dict[str, Any]:
        return {"cancelled": None}


State = Union[Queued, Running, Completed, Failed, Rejected, Cancelled]


def state_from_json(raw: dict[str, Any]) -> State:
    """按单键对象的键名还原 State；多键与未知键都拒绝。"""

    if len(raw) != 1:
        raise ValueError(f"State 必须是单键对象，得到 {sorted(raw)}")
    key, value = next(iter(raw.items()))
    if key == "queued":
        return Queued(queue=QueueProgress(**value))
    if key == "running":
        return Running()
    if key == "completed":
        return Completed()
    if key == "failed":
        return Failed(error=Failure.from_json(value))
    if key == "rejected":
        return Rejected(error=Failure.from_json(value))
    if key == "cancelled":
        return Cancelled()
    raise ValueError(f"未知状态键: {key!r}")


# ---------------------------------------------------------------------------
# SLI 与 SLO
# ---------------------------------------------------------------------------


# 六个 SLI 的数值类型。底层都是 float，类型名承担语义与单位。
CheckPointDurationMetric = float  # 一次 checkpoint 从开始写到元数据可见，秒
TrainingStallMetric = float       # 训练被写入阻塞的时长，秒
RestoreDurationMetric = float     # 从 checkpoint 恢复到可继续训练，秒
WriteThroughputMetric = float     # 写入期间按 rank 聚合的吞吐，字节/秒
MetadataCommitMetric = float      # 分片元数据提交到可见，毫秒
WriteLatencyP99Metric = float     # 单个写请求 P99，毫秒


@dataclass(frozen=True)
class CheckPointDuration:
    tag: ClassVar[str] = "check_point_duration_sli"
    value: CheckPointDurationMetric = 0.0

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.value}


@dataclass(frozen=True)
class TrainingStall:
    tag: ClassVar[str] = "training_stall_sli"
    value: TrainingStallMetric = 0.0

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.value}


@dataclass(frozen=True)
class RestoreDuration:
    tag: ClassVar[str] = "restore_duration_sli"
    value: RestoreDurationMetric = 0.0

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.value}


@dataclass(frozen=True)
class WriteThroughput:
    tag: ClassVar[str] = "write_throughput_sli"
    value: WriteThroughputMetric = 0.0

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.value}


@dataclass(frozen=True)
class MetadataCommit:
    tag: ClassVar[str] = "metadata_commit_sli"
    value: MetadataCommitMetric = 0.0

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.value}


@dataclass(frozen=True)
class WriteLatencyP99:
    tag: ClassVar[str] = "write_latency_p99_sli"
    value: WriteLatencyP99Metric = 0.0

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.value}


CkptSli = Union[
    CheckPointDuration,
    TrainingStall,
    RestoreDuration,
    WriteThroughput,
    MetadataCommit,
    WriteLatencyP99,
]

_SLI_BY_KEY = {
    "check_point_duration_sli": CheckPointDuration,
    "training_stall_sli": TrainingStall,
    "restore_duration_sli": RestoreDuration,
    "write_throughput_sli": WriteThroughput,
    "metadata_commit_sli": MetadataCommit,
    "write_latency_p99_sli": WriteLatencyP99,
}


def ckpt_sli_from_json(raw: dict[str, Any]) -> CkptSli:
    if len(raw) != 1:
        raise ValueError(f"SLI 必须是单键对象，得到 {sorted(raw)}")
    key, value = next(iter(raw.items()))
    cls = _SLI_BY_KEY.get(key)
    if cls is None:
        raise ValueError(f"未知 SLI 键: {key!r}")
    return cls(value=value)


@dataclass(frozen=True)
class MetricLE:
    """actual 不超过 threshold 即通过。"""

    actual: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.actual <= self.threshold

    def to_json(self) -> dict[str, Any]:
        return {"actual": self.actual, "threshold": self.threshold}


@dataclass(frozen=True)
class MetricGE:
    """actual 不低于 threshold 即通过。"""

    actual: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.actual >= self.threshold

    def to_json(self) -> dict[str, Any]:
        return {"actual": self.actual, "threshold": self.threshold}


# 每个 SLO 绑定一个 SLI 类型；键名带 _slo 后缀。
@dataclass(frozen=True)
class CkptSloCheckpointDuration:
    tag: ClassVar[str] = "check_point_duration_slo"
    verdict: MetricLE = field(default_factory=lambda: MetricLE(0.0, 0.0))

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.verdict.to_json()}


@dataclass(frozen=True)
class CkptSloTrainingStall:
    tag: ClassVar[str] = "training_stall_slo"
    verdict: MetricLE = field(default_factory=lambda: MetricLE(0.0, 0.0))

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.verdict.to_json()}


@dataclass(frozen=True)
class CkptSloRestoreDuration:
    tag: ClassVar[str] = "restore_duration_slo"
    verdict: MetricLE = field(default_factory=lambda: MetricLE(0.0, 0.0))

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.verdict.to_json()}


@dataclass(frozen=True)
class CkptSloWriteThroughput:
    tag: ClassVar[str] = "write_throughput_slo"
    verdict: MetricGE = field(default_factory=lambda: MetricGE(0.0, 0.0))

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.verdict.to_json()}


@dataclass(frozen=True)
class CkptSloMetadataCommit:
    tag: ClassVar[str] = "metadata_commit_slo"
    verdict: MetricLE = field(default_factory=lambda: MetricLE(0.0, 0.0))

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.verdict.to_json()}


@dataclass(frozen=True)
class CkptSloWriteLatencyP99:
    tag: ClassVar[str] = "write_latency_p99_slo"
    verdict: MetricLE = field(default_factory=lambda: MetricLE(0.0, 0.0))

    def to_json(self) -> dict[str, Any]:
        return {self.tag: self.verdict.to_json()}


CkptSlo = Union[
    CkptSloCheckpointDuration,
    CkptSloTrainingStall,
    CkptSloRestoreDuration,
    CkptSloWriteThroughput,
    CkptSloMetadataCommit,
    CkptSloWriteLatencyP99,
]

_SLO_LE_BY_KEY = {
    "check_point_duration_slo": CkptSloCheckpointDuration,
    "training_stall_slo": CkptSloTrainingStall,
    "restore_duration_slo": CkptSloRestoreDuration,
    "metadata_commit_slo": CkptSloMetadataCommit,
    "write_latency_p99_slo": CkptSloWriteLatencyP99,
}
_SLO_GE_BY_KEY = {"write_throughput_slo": CkptSloWriteThroughput}


def ckpt_slo_from_json(raw: dict[str, Any]) -> CkptSlo:
    if len(raw) != 1:
        raise ValueError(f"SLO 必须是单键对象，得到 {sorted(raw)}")
    key, value = next(iter(raw.items()))
    if key in _SLO_LE_BY_KEY:
        return _SLO_LE_BY_KEY[key](verdict=MetricLE(**value))
    if key in _SLO_GE_BY_KEY:
        return _SLO_GE_BY_KEY[key](verdict=MetricGE(**value))
    raise ValueError(f"未知 SLO 键: {key!r}")


# ---------------------------------------------------------------------------
# 产物与运行请求
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifacts:
    """一次运行写完的四份产物。未产出的字段用空字符串占位；schema 要求全必填。"""

    manifest: str = ""
    metrics: tuple[CkptSli, ...] = ()
    slo: tuple[CkptSlo, ...] = ()
    report: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "metrics": [m.to_json() for m in self.metrics],
            "slo": [s.to_json() for s in self.slo],
            "report": self.report,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Artifacts":
        _only(raw, ('manifest', 'metrics', 'slo', 'report'), "Artifacts")
        return cls(
            manifest=raw["manifest"],
            metrics=tuple(ckpt_sli_from_json(item) for item in raw["metrics"]),
            slo=tuple(ckpt_slo_from_json(item) for item in raw["slo"]),
            report=raw["report"],
        )


@dataclass(frozen=True)
class RunRecord:
    """服务返回给调用方的运行标识、状态、进度与产物位置。"""

    run_id: str
    state: State
    progress: WorkProgress = field(default_factory=lambda: WorkProgress(0, 0))
    artifacts: Artifacts = field(default_factory=Artifacts)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state.to_json(),
            "progress": self.progress.to_json(),
            "artifacts": self.artifacts.to_json(),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "RunRecord":
        _only(raw, ('run_id', 'state', 'progress', 'artifacts'), "RunRecord")
        return cls(
            run_id=raw["run_id"],
            state=state_from_json(raw["state"]),
            progress=WorkProgress(**raw["progress"]),
            artifacts=Artifacts.from_json(raw["artifacts"]),
        )


@dataclass(frozen=True)
class RunSpec:
    """提交给样本服务的一次运行请求。"""

    sample: str       # name@version
    config: str       # 样本 repo 里那份 SampleConfigBase 的名字
    storage: StorageLayer
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "config": self.config,
            "storage": self.storage.to_json(),
            "seed": self.seed,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "RunSpec":
        _only(raw, ('sample', 'config', 'storage', 'seed'), "RunSpec")
        return cls(
            sample=raw["sample"],
            config=raw["config"],
            storage=StorageLayer.from_json(raw["storage"]),
            seed=raw["seed"],
        )
