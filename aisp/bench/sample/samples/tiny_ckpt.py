"""tiny-ckpt@0.1.0：一个不依赖 GPU / torch 的 checkpoint 样本。

按旁路原则：不跑真实模型前向，只按真实分片策略把字节写下去，再冷链读回来。
用线程池模拟多 rank，单进程内并发。产出 6 种 CkptSli。

规模由 SampleConfigBase.topology 决定：rank 数 = nodes * gpus_per_node。
每 rank 分片大小固定（默认 4 MiB），便于跨规模比较。

两条测量上的硬约束：
- 恢复必须冷读：读前对文件做 POSIX_FADV_DONTNEED，否则测到的是页缓存而不是存储。
- 工作目录跟随 mount：nfs / fuse 必须由 TINY_CKPT_ROOT 指向真实挂载点，
  不给就报错，不允许默默写本地盘。
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..configs import SampleConfigBase
from ..types import (
    CheckPointDuration,
    MetadataCommit,
    RestoreDuration,
    RunSpec,
    TrainingStall,
    WriteLatencyP99,
    WriteThroughput,
)

BYTES_PER_RANK = 4 * 1024 * 1024  # 4 MiB / rank
WRITE_CHUNK = 64 * 1024           # 64 KiB 一次 write()，用于测 P99
REMOTE_MOUNTS = ("nfs", "fuse")


def _workdir(config: SampleConfigBase) -> Path:
    """按 mount 决定工作目录；远端挂载必须显式给根路径。"""

    root = os.environ.get("TINY_CKPT_ROOT")
    if root:
        return Path(tempfile.mkdtemp(prefix="tiny-ckpt-", dir=root))
    if config.topology.mount in REMOTE_MOUNTS:
        raise RuntimeError(
            f"mount={config.topology.mount} 需要 TINY_CKPT_ROOT 指向真实挂载点，"
            "否则测到的是本地盘"
        )
    return Path(tempfile.mkdtemp(prefix="tiny-ckpt-"))


def _write_one_rank(
    rank_id: int,
    directory: Path,
    buf: bytes,
    write_latencies_ms: list[float],
    latencies_lock: threading.Lock,
) -> int:
    """一个 rank 写自己的分片，复用调用方预生成的 buf。

    buf 预生成后写路径上不再调 os.urandom，否则 write 延迟里会混入随机数生成。
    """

    target = directory / f"rank-{rank_id:05d}.bin"
    written = 0
    with open(target, "wb", buffering=0) as handle:
        for off in range(0, len(buf), WRITE_CHUNK):
            piece = buf[off:off + WRITE_CHUNK]
            t0 = time.perf_counter()
            handle.write(piece)
            t1 = time.perf_counter()
            with latencies_lock:
                write_latencies_ms.append((t1 - t0) * 1000.0)
            written += len(piece)
        os.fsync(handle.fileno())
    return written


def _drop_pages(handle) -> None:
    """丢弃该文件的页缓存，让后续读走存储。不支持的平台静默跳过。"""

    try:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _read_all_ranks(directory: Path, rank_count: int) -> int:
    """冷读全部分片：每个文件读前先丢页缓存。"""

    total = 0
    for rank_id in range(rank_count):
        target = directory / f"rank-{rank_id:05d}.bin"
        with open(target, "rb", buffering=0) as handle:
            _drop_pages(handle)
            while handle.read(WRITE_CHUNK):
                pass
        total += target.stat().st_size
    return total


def _commit_metadata(directory: Path, rank_count: int) -> None:
    """把每个分片文件的元数据刷到可见，再写并 fsync 运行清单。"""

    for rank_id in range(rank_count):
        with open(directory / f"rank-{rank_id:05d}.bin", "rb+") as handle:
            os.fsync(handle.fileno())

    manifest = directory / "manifest.json"
    with open(manifest, "wb", buffering=0) as handle:
        handle.write(f'{{"ranks": {rank_count}}}\n'.encode())
        handle.flush()
        os.fsync(handle.fileno())


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(len(ordered) * 0.99)))
    return ordered[idx]


def runner(spec: RunSpec, config: SampleConfigBase) -> list:
    # Topology 已保证 nodes / gpus_per_node 至少为 1；这里不再用 max(1, ...)
    # 掩盖 0，否则 0 卡配置会静默跑成 1 rank 并产出看似正常的指标。
    rank_count = config.topology.nodes * config.topology.gpus_per_node
    workdir = _workdir(config)
    latencies: list[float] = []
    lock = threading.Lock()
    # 一次性预生成分片内容；写路径上不再调 os.urandom，p99 才是纯写延迟。
    shard_buf = os.urandom(BYTES_PER_RANK)

    try:
        # ---- 同步写入：每个 rank 一个线程，不做并发度封顶 ----
        t_write_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=rank_count) as pool:
            list(
                pool.map(
                    lambda r: _write_one_rank(r, workdir, shard_buf, latencies, lock),
                    range(rank_count),
                )
            )
        t_write_end = time.perf_counter()

        # ---- 元数据提交：分片 fsync + 清单 fsync ----
        t_meta_start = time.perf_counter()
        _commit_metadata(workdir, rank_count)
        t_meta_end = time.perf_counter()

        # ---- 恢复：冷读全部分片 ----
        t_restore_start = time.perf_counter()
        _read_all_ranks(workdir, rank_count)
        t_restore_end = time.perf_counter()
    finally:
        for path in workdir.iterdir():
            path.unlink()
        workdir.rmdir()

    write_duration = t_write_end - t_write_start
    total_written = rank_count * BYTES_PER_RANK
    return [
        CheckPointDuration(value=write_duration),
        # 同步写入：训练停顿等于写入本身。异步重叠的场景不在本样本范围内。
        TrainingStall(value=write_duration),
        RestoreDuration(value=t_restore_end - t_restore_start),
        WriteThroughput(value=total_written / write_duration if write_duration > 0 else 0.0),
        MetadataCommit(value=(t_meta_end - t_meta_start) * 1000.0),
        WriteLatencyP99(value=_p99(latencies)),
    ]
