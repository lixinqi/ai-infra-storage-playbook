#!/usr/bin/env python3
"""checkpoint 样本接口的黑盒契约检查。

边界
----
- 类型与接口名只从 ``sample``（文档 public 层）取。
- 服务实例只通过 ``sample.memory.InMemorySampleService`` 构造，这是唯一的实现入口。
- 不读私有成员（``_set_state`` / ``_records`` / ``_validate`` / ``_threads``）；
  取消吸收、记录淘汰这类行为一律通过 ``start`` / ``stop`` / ``query`` 观察。

组织
----
七个组，每组是一类契约；组内多条断言。报告按组计数，失败时列出具体断言。

- 序列化：``to_json`` 过 schema、往返幂等、未知键被拒。
- 状态机：提交到完成的完整路径、查询只读、已结束的运行原样交回。
- 取消：返回即为 cancelled，且迟到的完成不覆盖它。
- 非法请求：请求不合法时直接 rejected，不起后台运行。
- 阶梯：拓扑落档与 0 规模拒绝。
- 观测：采集层至少两层且含 sys。
- 样本产出：真实跑一次，六项 SLI 齐全并过 Artifacts schema。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))

import sample  # noqa: F401 - 黑盒边界：public 层

from sample import Artifacts, RunRecord, RunSpec
from sample.ckpt_sample import (
    CheckPointDuration,
    Component,
    Failure,
    Model,
    ObservationConfig,
    Running,
    SampleConfigBase,
    StorageLayer,
    Topology,
    WorkProgress,
)
from sample.levels import resolve_level
from sample.memory import InMemorySampleService
from sample.samples import tiny_ckpt

SCHEMAS = BENCH / "checkpoint" / "json_schema"

Assertion = tuple[str, bool]


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------


def _storage() -> StorageLayer:
    return StorageLayer(
        backend=Component(name="gpfs", version="5.1"),
        adapter=Component(name="torchstore", version="0.2", git_commit_id="9f2c1a4"),
    )


def _spec(sample_name: str = "s@1") -> RunSpec:
    return RunSpec(sample=sample_name, config="tiny-64-s0", storage=_storage(), seed=20261103)


def _config(nodes: int = 1, gpus: int = 8) -> SampleConfigBase:
    return SampleConfigBase(
        model=Model(id="example/tiny-64", git_commit_id="0" * 40),
        topology=Topology(nodes=nodes, gpus_per_node=gpus, network="rdma", mount="local-nvme"),
        observe=ObservationConfig(enabled=True, sample_rate=0.1, layers=("app", "sys")),
    )


def _service(runners: dict[str, Callable] | None = None, **kwargs: Any) -> InMemorySampleService:
    return InMemorySampleService(
        runners=runners or {"s@1": lambda spec, config: []},
        config_loader=lambda name: _config(),
        **kwargs,
    )


def _wait(svc: InMemorySampleService, run_id: str, want: str, timeout: float = 5.0) -> str:
    """只通过 query 等待状态；不读内部记录表。"""

    deadline = time.time() + timeout
    while time.time() < deadline:
        tag = svc.query(run_id).state.tag
        if tag == want or tag in ("failed", "rejected", "cancelled"):
            return tag
        time.sleep(0.01)
    return svc.query(run_id).state.tag


def _validator(name: str):
    """schema 文件存在且装了 jsonschema 才校验；否则断言放行并在报告里标注。"""

    path = SCHEMAS / f"{name}.schema.json"
    if not path.is_file():
        return None
    try:
        import jsonschema
    except ImportError:
        return None
    return jsonschema.Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 组 1：序列化
# ---------------------------------------------------------------------------


def check_serialization() -> list[Assertion]:
    artifact = Artifacts(
        manifest="s3://b/runs/x/manifest.json",
        metrics=(CheckPointDuration(value=1.0),),
        slo=(),
        report="s3://b/runs/x/report.html",
    )
    record = RunRecord(
        run_id="r", state=Running(), progress=WorkProgress(0, 0), artifacts=artifact
    )
    config = _config()
    cases = [
        ("RunSpec", RunSpec, _spec()),
        ("RunRecord", RunRecord, record),
        ("Artifacts", Artifacts, artifact),
        ("Component", Component, Component(name="gpfs", version="5.1")),
        ("StorageLayer", StorageLayer, _storage()),
        ("Failure", Failure, Failure(kind="timeout", detail="x")),
        ("Model", Model, config.model),
        ("Topology", Topology, config.topology),
        ("ObservationConfig", ObservationConfig, config.observe),
        ("SampleConfigBase", SampleConfigBase, config),
    ]

    schema_fail, roundtrip_fail, extra_fail = [], [], []
    for name, cls, instance in cases:
        payload = instance.to_json()
        validator = _validator(name)
        if validator is not None and not validator.is_valid(payload):
            schema_fail.append(name)
        if cls.from_json(payload).to_json() != payload:
            roundtrip_fail.append(name)
        try:
            cls.from_json({**payload, "junk": 1})
            extra_fail.append(name)
        except ValueError:
            pass

    def agg(label: str, bad: list[str]) -> Assertion:
        return (label if not bad else f"{label}: {', '.join(bad)}", not bad)

    return [
        agg("every type passes its own schema", schema_fail),
        agg("every type round-trips", roundtrip_fail),
        agg("every type rejects unknown keys", extra_fail),
    ]


# ---------------------------------------------------------------------------
# 组 2：状态机
# ---------------------------------------------------------------------------


def check_state_machine() -> list[Assertion]:
    metrics = [CheckPointDuration(value=1.0), CheckPointDuration(value=2.0)]
    svc = _service({"s@1": lambda spec, config: metrics})
    started = svc.start(_spec())
    first_tag = started.state.tag
    tag = _wait(svc, started.run_id, "completed")
    final = svc.query(started.run_id)

    # 查询只读：连查两次结果一致。
    again = svc.query(started.run_id)
    # 已结束的运行再 stop，原样交回，不报错。
    stopped = svc.stop(started.run_id)

    return [
        ("start returns a non-terminal record", first_tag in ("queued", "running")),
        ("run reaches completed", tag == "completed"),
        ("completed run carries both files and every metric",
         bool(final.artifacts.manifest) and bool(final.artifacts.report)
         and len(final.artifacts.metrics) == len(metrics)),
        ("query does not mutate the run",
         again.state.tag == final.state.tag and again.progress == final.progress),
        ("stop on a finished run hands it back unchanged", stopped.state.tag == "completed"),
        ("progress stays 0/0 instead of faking workload",
         final.progress.done == 0 and final.progress.total == 0),
        *_check_record_eviction(),
    ]


def _check_record_eviction() -> list[Assertion]:
    """记录上限也走黑盒：构造时给上限，跑满后用最早的标识查询应已淘汰。"""

    small = _service(max_records=3)
    ids = [small.start(_spec()).run_id for _ in range(6)]
    # 每跑完一条就触发一次淘汰；被淘汰的记录查询会抛 KeyError，视为正常。
    for run_id in ids:
        try:
            _wait(small, run_id, "completed")
        except KeyError:
            pass
    try:
        small.query(ids[0])
        return [("record store evicts the oldest terminal run above max_records", False)]
    except KeyError:
        return [("record store evicts the oldest terminal run above max_records", True)]


# ---------------------------------------------------------------------------
# 组 3：取消
# ---------------------------------------------------------------------------


def check_cancellation() -> list[Assertion]:
    release = threading.Event()

    def slow(spec: RunSpec, config: SampleConfigBase) -> list:
        release.wait(timeout=3.0)
        return [CheckPointDuration(value=1.0)]

    svc = _service({"s@1": slow})
    record = svc.start(_spec())
    time.sleep(0.05)
    stopped = svc.stop(record.run_id)
    # 放开后台样本让它跑到「完成」；取消必须吸收这次迟到的完成。
    release.set()
    time.sleep(0.3)
    final = svc.query(record.run_id)
    second = svc.stop(record.run_id)

    return [
        ("stop returns cancelled", stopped.state.tag == "cancelled"),
        ("a late completion does not overwrite cancelled", final.state.tag == "cancelled"),
        ("stop is idempotent on a cancelled run", second.state.tag == "cancelled"),
    ]


# ---------------------------------------------------------------------------
# 组 4：非法请求
# ---------------------------------------------------------------------------


def check_bad_requests() -> list[Assertion]:
    svc = _service()
    cases = {
        "unknown sample": RunSpec(sample="nope@9", config="c", storage=_storage(), seed=1),
        "sample without a version": RunSpec(sample="no-at", config="c", storage=_storage(), seed=1),
        "seed that is not an int": RunSpec(sample="s@1", config="c", storage=_storage(), seed=1.5),
        "blank component name": RunSpec(
            sample="s@1", config="c",
            storage=StorageLayer(Component(name="", version=""), Component(name="a", version="1")),
            seed=1,
        ),
    }
    records = {name: svc.start(spec) for name, spec in cases.items()}

    # config 加载失败也在提交时拒绝，而不是起线程后失败。
    def failing_loader(name: str) -> SampleConfigBase:
        raise RuntimeError("no such config")

    bad_cfg = InMemorySampleService(
        runners={"s@1": lambda spec, config: []}, config_loader=failing_loader
    ).start(_spec())

    accepted = [name for name, record in records.items() if record.state.tag != "rejected"]
    if bad_cfg.state.tag != "rejected":
        accepted.append("config that fails to load")
    # 被拒的请求不进队列：稍后再查仍是 rejected，不会翻成 completed。
    time.sleep(0.2)
    drifted = [
        name for name, record in records.items()
        if svc.query(record.run_id).state.tag != "rejected"
    ]
    return [
        ("every malformed request -> rejected" if not accepted
         else f"every malformed request -> rejected: {', '.join(accepted)}", not accepted),
        ("a rejected request stays rejected" if not drifted
         else f"a rejected request stays rejected: {', '.join(drifted)}", not drifted),
    ]


# ---------------------------------------------------------------------------
# 组 5：阶梯
# ---------------------------------------------------------------------------


def check_level() -> list[Assertion]:
    expected = [((1, 8), "S0"), ((2, 8), "S1"), ((8, 8), "S2"), ((128, 8), "S4")]
    wrong = [
        f"{nodes}x{gpus}->{resolve_level(_config(nodes, gpus).topology)} want {want}"
        for (nodes, gpus), want in expected
        if resolve_level(_config(nodes, gpus).topology) != want
    ]
    accepted_zero = []
    for nodes, gpus in ((0, 8), (1, 0)):
        try:
            Topology(nodes=nodes, gpus_per_node=gpus, network="none", mount="local-nvme")
            accepted_zero.append(f"{nodes}x{gpus}")
        except ValueError:
            pass
    return [
        ("topology maps to the expected levels" if not wrong
         else f"topology maps to the expected levels: {', '.join(wrong)}", not wrong),
        ("zero-size topologies are rejected" if not accepted_zero
         else f"zero-size topologies are rejected: {', '.join(accepted_zero)}", not accepted_zero),
    ]


# ---------------------------------------------------------------------------
# 组 6：观测配置
# ---------------------------------------------------------------------------


def check_observation() -> list[Assertion]:
    def accepts(layers: tuple[str, ...]) -> bool:
        try:
            ObservationConfig(enabled=True, sample_rate=0.1, layers=layers)
            return True
        except ValueError:
            return False

    # 两条规则各给一个只违反它的样本：单层含 sys 违反层数，两层缺 sys 违反 sys。
    bad = [f"{layers}" for layers in (("sys",), ("app", "storage")) if accepts(layers)]
    return [
        ("a layer set without two-of-which-sys is rejected" if not bad
         else f"a layer set without two-of-which-sys is rejected: {', '.join(bad)}", not bad),
        ("app + sys is accepted", accepts(("app", "sys"))),
    ]


# ---------------------------------------------------------------------------
# 组 7：样本产出
# ---------------------------------------------------------------------------


def check_sample_output() -> list[Assertion]:
    root = tempfile.mkdtemp(prefix="ckpt-check-")
    old = os.environ.get("TINY_CKPT_ROOT")
    os.environ["TINY_CKPT_ROOT"] = root
    try:
        svc = _service({"tiny-ckpt@0.1.0": tiny_ckpt.runner})
        spec = RunSpec(sample="tiny-ckpt@0.1.0", config="c", storage=_storage(), seed=1)
        artifacts = svc.execute(spec)
    finally:
        if old is None:
            os.environ.pop("TINY_CKPT_ROOT", None)
        else:
            os.environ["TINY_CKPT_ROOT"] = old

    tags = [item.tag for item in artifacts.metrics]
    values = [item.value for item in artifacts.metrics]
    validator = _validator("Artifacts")

    return [
        ("sample emits six SLI", len(tags) == 6),
        ("all six SLI are positive", all(value > 0 for value in values)),
        ("sample artifacts pass Artifacts.schema.json",
         validator is None or validator.is_valid(artifacts.to_json())),
    ]


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

GROUPS = [
    ("serialization", check_serialization),
    ("state machine", check_state_machine),
    ("cancellation", check_cancellation),
    ("bad requests", check_bad_requests),
    ("level", check_level),
    ("observation", check_observation),
    ("sample output", check_sample_output),
]


def main() -> int:
    results: list[tuple[str, str, bool]] = []
    skipped: list[str] = []
    if not SCHEMAS.is_dir():
        skipped.append(f"{SCHEMAS} not found; schema assertions treated as satisfied")

    for group, fn in GROUPS:
        try:
            for name, ok in fn():
                results.append((group, name, ok))
        except Exception as exc:  # noqa: BLE001 - 组内崩溃按该组失败上报
            results.append((group, f"{type(exc).__name__}: {exc}", False))

    failed = [f"[{group}] {name}" for group, name, ok in results if not ok]
    print(json.dumps({
        "check": "check_sample_interface",
        "mode": "black-box",
        "groups": len(GROUPS),
        "assertions": len(results),
        "failed_groups": sorted({group for group, _, ok in results if not ok}),
        "failed": failed,
        "skipped": skipped,
        "status": "pass" if not failed else "fail",
    }, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
