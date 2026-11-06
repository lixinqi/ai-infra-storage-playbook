#!/usr/bin/env python3
"""命令行入口：读 RunSpec JSON + SampleConfigBase JSON，跑一次并落盘 Artifacts。

`--out` 写的是纯 Artifacts，可直接用 Artifacts.schema.json 校验。
等级不写进产物（schema 里没有这个字段），打到 stderr。

用法：
    python3 -m sample.runner_cli --spec spec.json --config config.json --out artifacts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import RunSpec, SampleConfigBase
from .memory import InMemorySampleService
from .samples import tiny_ckpt


def load_spec(path: Path) -> RunSpec:
    return RunSpec.from_json(json.loads(path.read_text(encoding="utf-8")))


def load_config(path: Path) -> SampleConfigBase:
    return SampleConfigBase.from_json(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    config = load_config(args.config)

    svc = InMemorySampleService(
        runners={"tiny-ckpt@0.1.0": tiny_ckpt.runner},
        config_loader=lambda name: config,
    )
    level = svc.get_level(config)
    artifacts = svc.execute(spec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifacts.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "level": level, "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
