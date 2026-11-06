# sample：checkpoint 样本接口的 Python 落地

对齐 [`checkpoint/ckpt-sample-interface.md`](../checkpoint/ckpt-sample-interface.md)。
一次运行 = 一条样本 × 一个存储版本；一份 `RunSpec` 进，一条 `RunRecord` 出。

每个导出类型的 `to_json()` 输出，直接通过 [`checkpoint/json_schema/`](../checkpoint/json_schema/) 的同名 schema。

## 模块

| 文件 | 对应文档模块 | 是什么 |
|---|---|---|
| `ckpt_sample.py` | `ckpt_sample` | 聚合层，按文档模块视图导出全部类型；值类型在 `types.py`，配置在 `configs.py` |
| `types.py` | — | RunSpec / RunRecord / StorageLayer / Component / State(oneof) / QueueProgress / WorkProgress / Artifacts / CkptSli(6 种) / CkptSlo(MetricLE/GE) / 6 个 Metric / Failure |
| `configs.py` | — | SampleConfigBase（model + topology + observe）；Model / Topology / ObservationConfig |
| `levels.py` | `levels`（checkpoint/scaling.md） | `Level` = S0..S5；默认阶梯与 `resolve_level` |
| `observation.py` | `observation`（checkpoint/methodology.md） | `ObservationLayer` = app / sys / storage |
| `ckpt_sample_api.py` | `ckpt_sample_api` | `SampleService` Protocol：Start / Stop / Query / Execute / GetLevel；`SampleRunner` 是样本作者要实现的 callable |
| `memory.py` | — | `InMemorySampleService` 参考实现：后台线程跑 Start、锁内推进状态机、可取消 |
| `__init__.py` | `public` | 只做转发：`import ckpt_sample`、`import ckpt_sample_api`，再绑定十一个名字 |

## JSON 形态

oneof 是单键对象，键名就是 schema 里的属性名：

```json
{"state": {"queued": {"pos": 1, "len": 3}}}
{"state": {"completed": null}}
{"metrics": [{"check_point_duration_sli": 0.064}, {"write_throughput_sli": 5.2e8}]}
{"slo": [{"check_point_duration_slo": {"actual": 0.064, "threshold": 0.1}}]}
```

`RunRecord` 的四个字段与 `Artifacts` 的四个字段在 schema 里都是必填；未产出的路径用空字符串占位。

## 最小用法

```python
from sample import RunSpec, SampleConfigBase
from sample.memory import InMemorySampleService
from sample.types import Component, StorageLayer, CheckPointDuration

def my_sample(spec, config):
    return [CheckPointDuration(value=12.5)]  # 六种 CkptSli 任选

svc = InMemorySampleService(
    runners={"tiny-ckpt@0.1.0": my_sample},
    config_loader=lambda name: load_my_config(name),
)

record = svc.start(RunSpec(
    sample="tiny-ckpt@0.1.0",
    config="tiny-64-s0",
    storage=StorageLayer(
        backend=Component(name="gpfs", version="5.1"),
        adapter=Component(name="torchstore", version="0.2"),
    ),
    seed=20261103,
))
record.to_json()  # 直接可过 RunRecord.schema.json
```

## 自检

```text
python3 checks/check_sample_interface.py
```

黑盒契约检查：类型与接口名只从 `sample` 的 public 层取，服务实例只经 `sample.memory.InMemorySampleService` 构造，不读 `_set_state` / `_records` / `_threads`。七组、22 条断言；失败按组上报并列出具体项。

| 组 | 覆盖 |
|---|---|
| serialization | 每个类型 `to_json` 过同名 schema、往返幂等、拒未知键 |
| state machine | 提交到完成、查询只读、已结束的运行原样交回、`progress` 恒 0/0、记录上限淘汰 |
| cancellation | 返回即 `cancelled`，迟到的完成不覆盖 |
| bad requests | 未知样本 / 缺版本 / 非 int seed / 空白组件 / config 加载失败 → `rejected` |
| level | 四档落档，0 规模拓扑被拒 |
| observation | 层数规则与 sys 规则各给一个只违反它的样本 |
| sample output | 六项 SLI 全为正，过 `Artifacts.schema.json` |

`checkpoint/json_schema` 缺失或不装 `jsonschema` 时，schema 断言放行并在 `skipped` 字段里写明，不冒充已校验。

变异测试：改坏 7 处实现，6 处被抓。`_set_state` 的取消吸收分支没被抓，因为 `_run_background` 已先在 `cancel_event` 上拦住迟到的完成，这一层在黑盒下不可观测，属等价变异体。

## 测量口径

- **恢复是冷读**：读每个分片前先 `POSIX_FADV_DONTNEED` 丢页缓存，否则测到的是页缓存而不是存储。热读与冷读在同一台 64 核机上差 17–26 倍。
- **工作目录跟随 mount**：`nfs` / `fuse` 必须用 `TINY_CKPT_ROOT` 指向真实挂载点，不给就报错，不允许默默写本地盘。`local-nvme` 走临时目录。
- **写入是同步模式**：`training_stall_sli` 因此等于 `check_point_duration_sli`。异步写入与计算重叠的场景不在本样本范围内。
- **元数据提交**测的是「全部分片 fsync + 清单 fsync」，不是单个文件的 fsync。

## 边界

- 这是**样本接口**的落地，不是评估服务的 HTTP 层。HTTP/CLI 适配在 `checkpoint/evaluation-service.md`，后续再做。
- `InMemorySampleService` 只用于开发期自检：不做进程隔离，产物路径是 `memory://` 占位，生产实现要替换 `artifact_writer` 落对象存储。
- S0–S5 阶梯的默认阈值在 `levels.py`；配方要覆盖时由 `get_level` 注入自己的阶梯，不在代码里写死。
- `Topology` 的 `network` / `mount` 用 `Literal` 收窄，比 schema 的 `string` 更严；schema 只约束类型，取值约束来自文档 Hint。
- `ObservationConfig` 开启观测时要求 `layers` 至少两层且包含 `sys`，这是文档对交叉核对的要求。
- `Topology` 要求 `nodes` 与 `gpus_per_node` 至少为 1。schema 只写了 `integer`，没有下限；0 规模算不出有意义的档位，样本侧也会退化成空跑，因此解码与构造都拒绝。
- `start()` 只登记请求就返回；`stop()` 是取消请求，返回时状态已是 `cancelled`，但样本本身可能还跑到下一个检查点。`cancelled` 是吸收态，后台线程迟到的 `completed` / `failed` 不会覆盖它。
- 记录默认最多留 1000 条（`max_records`），超出时按插入顺序淘汰最早的终态记录；被淘汰的 `run_id` 再 `query` 会抛 `KeyError`。运行中的记录不淘汰。
- `progress` 恒为 0/0。文档对 `WorkProgress.total` 的原话是「总工作量；还没算出来时是 0」，0 是这个字段的合法取值；`SampleRunner` 的返回类型只有 `list[CkptSli]`，没有进度回调通道，服务层拿不到中间量。所以这里报「未知」，不拿 SLI 条数冒充 `total`，完成与否只看 `state`。
- 运行中的 `Artifacts` 用四个空值占位（`""` / `[]`），schema 里四个字段都是 required 且没有 nullable，只能这样表达「尚未产出」；已验证空值形态通过 `Artifacts.schema.json`。详见下面「与文档的差异」。
- `public` 层按文档转发：`import ckpt_sample`、`import ckpt_sample_api`，再绑定十一个名字（Start / Stop / Query / Execute / GetLevel / RunSpec / RunRecord / Artifacts / CkptSli / CkptSlo / SampleConfigBase）。实现类要从 `sample.memory`、`sample.types`、`sample.levels` 导入。
- `Start` / `Stop` 等是 `SampleService` 的未绑定方法，使用时要带实例：`Start(svc, spec)`。

## 与文档的差异

| 项 | 文档 | schema / 实现 | 处理 |
|---|---|---|---|
| `Artifacts` 未完成形态 | 「没跑完时相应字段缺」 | schema 四字段全 required、无 nullable | 实现选 schema 侧，用空值占位；冲突待上游决定放宽 schema 还是改措辞 |
| `Topology.network` / `mount` | 明确列 `rdma / tcp / none` 与 `local-nvme / nfs / fuse` | schema 只写 `string`，无 enum | 实现按文档用 `Literal` 收窄；建议给 schema 补 enum |
| `Failure.kind` 的拒绝类取值 | 「environment、dependency、timeout、observation、assertion 等」，以「等」结尾 | schema 只写 `string` | `rejected` 必须带 `Failure`，文档未给拒绝类分类，实现取 `request`，落在「等」的开放集合内 |
